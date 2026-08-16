"""Registered orchestration for an append-only CN Fundamental successor.

The ordinary ``fundamental-maintain --allow-live`` path is intentionally not
used here.  This module freezes the current Fundamental, market and PIT
pointers before provider acquisition, validates a target-window market keyset,
creates an isolated support fileset, derives an append-only successor, and
writes only an isolated staging generation.  Canonical publication belongs to
the separate successor-promotion capability.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterator, Mapping, NoReturn, Sequence

import pyarrow.parquet as pq

from .fundamental_generation import (
    FUNDAMENTAL_POINTER_FILENAME,
    FUNDAMENTAL_TABLES,
    load_fundamental_pointer,
)
from .fundamental_incremental import (
    SUCCESSOR_APPEND_FIRST_MODE,
    SUCCESSOR_FINANCIAL_DEPENDENCY_CONTRACT_SHA256,
    assemble_safe_successor,
    build_successor_chain,
    build_keyset_closure,
    capture_parent_closure,
    seal_successor_provider_manifest,
    seal_support_plan,
    stage_successor_generation,
)
from .fundamental_historical_taint import (
    build_historical_taint_registry,
)
from .fundamental_provider_contract import canonical_json_sha256
from .fundamental_successor_source import (
    acquire_successor_support,
    build_successor_support_plan,
    open_support_tables,
    successor_support_evidence_paths,
    validate_successor_capture_fileset,
    validate_successor_support_fileset,
)
from .fundamental_taint import analyze_deferred_fundamental_taints


SAFE_SUCCESSOR_MAINTENANCE_SCHEMA = (
    "cn-fundamental-safe-successor-maintenance.v1"
)
EXPECTED_HISTORY_AUDIT_SCHEMA = "myquant-cn-history-audit.v4"
SAFE_SUCCESSOR_RESOURCE_PREFLIGHT_SCHEMA = (
    "cn-fundamental-safe-successor-resource-preflight.v2"
)
TAINT_DRY_RUN_EXECUTION_SCHEMA = "cn-fundamental-taint-dry-run-execution.v1"
_TS_CODE_RE = re.compile(r"^[0-9]{6}\.(?:BJ|SH|SZ)$", re.ASCII)


class FundamentalSuccessorMaintenanceError(RuntimeError):
    """One fail-closed safe-successor orchestration blocker."""

    def __init__(self, code: str, message: str = "") -> None:
        self.code = str(code)
        super().__init__(f"{self.code}: {message}" if message else self.code)


def _fail(code: str, message: str = "") -> NoReturn:
    raise FundamentalSuccessorMaintenanceError(code, message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sealed(value: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    body = dict(value)
    if field in body:
        _fail("RECEIPT_FIELD_COLLISION", field)
    body[field] = _sha256(_canonical_json_bytes(body))
    return body


def _private_audit_root(path: str | Path, *, workspace_root: Path) -> Path:
    root = Path(path).expanduser()
    if not root.is_absolute():
        _fail("TAINT_AUDIT_ROOT_MUST_BE_ABSOLUTE")
    root = root.absolute()
    if root == workspace_root or workspace_root in root.parents:
        _fail("TAINT_AUDIT_ROOT_MUST_BE_ISOLATED")
    if root.exists():
        metadata = os.lstat(root)
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or metadata.st_nlink < 2
        ):
            _fail("TAINT_AUDIT_ROOT_UNSAFE")
        allowed = {
            "analysis",
            "capture",
            "execution_receipt.json",
            "frozen_state.json",
        }
        if not {entry.name for entry in root.iterdir()}.issubset(allowed):
            _fail("TAINT_AUDIT_ROOT_ENTRY_SET_INVALID")
        receipt = root / "execution_receipt.json"
        if receipt.exists():
            payload = _json_object(
                _stable_regular_bytes(receipt, label="taint terminal receipt"),
                label="taint terminal receipt",
            )
            if payload.get("taint_analysis_status") != "PASS":
                _fail("TAINT_BLOCKED_RUN_NOT_RESUMABLE")
            _fail("TAINT_TERMINAL_RUN_ID_REUSED")
    else:
        if not root.parent.exists() or not root.parent.is_dir():
            _fail("TAINT_AUDIT_PARENT_MISSING")
        root.mkdir(mode=0o700)
    os.chmod(root, 0o700)
    return root


def _atomic_private_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical_json_bytes(value)
    if path.exists():
        if _stable_regular_bytes(path, label=f"audit {path.name}") != payload:
            _fail("TAINT_AUDIT_IMMUTABLE_CONFLICT", path.name)
        return
    descriptor = -1
    temporary = ""
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.",
            dir=path.parent,
        )
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = ""
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary:
            try:
                os.unlink(temporary)
            except OSError:
                pass
    readback = _stable_regular_bytes(path, label=f"audit {path.name}")
    if readback != payload:
        _fail("TAINT_AUDIT_READBACK_MISMATCH", path.name)


def _verify_authority_freeze(
    values: Mapping[Path, bytes],
    *,
    phase: str,
) -> dict[str, str]:
    identities: dict[str, str] = {}
    for path, expected in values.items():
        observed = _stable_regular_bytes(path, label=f"{phase} {path.name}")
        if observed != expected:
            _fail("BLOCKED_AUTHORITY_DRIFT", f"{phase}:{path}")
        identities[str(path)] = _sha256(observed)
    return identities


def _verify_file_freeze(
    values: Mapping[Path, tuple[str, int]],
    *,
    phase: str,
) -> dict[str, str]:
    identities: dict[str, str] = {}
    for path, expected in values.items():
        observed = _stable_regular_file_sha256(
            path,
            label=f"{phase} {path.name}",
        )
        if observed != expected:
            _fail("BLOCKED_AUTHORITY_DRIFT", f"{phase}:{path}")
        identities[str(path)] = observed[0]
    return identities


def _canonical_symbol(value: Any, *, label: str) -> str:
    if type(value) is not str or _TS_CODE_RE.fullmatch(value) is None:
        _fail("SYMBOL_IDENTITY_INVALID", label)
    return value


def _file_signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(stat.S_IFMT(value.st_mode)),
    )


def _stable_regular_bytes(path: str | Path, *, label: str) -> bytes:
    candidate = Path(path).expanduser()
    lexical = Path(
        os.path.abspath(
            os.fspath(candidate if candidate.is_absolute() else Path.cwd() / candidate)
        )
    )
    before = os.lstat(lexical)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        _fail("UNSAFE_AUTHORITY_FILE", label)
    try:
        descriptor = os.open(
            lexical,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise FundamentalSuccessorMaintenanceError(
            "AUTHORITY_READ_FAILED", label
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if _file_signature(opened) != _file_signature(before):
            _fail("AUTHORITY_CHANGED_DURING_OPEN", label)
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
        after = os.lstat(lexical)
        if (
            _file_signature(after) != _file_signature(before)
            or len(payload) != before.st_size
        ):
            _fail("AUTHORITY_CHANGED_DURING_READ", label)
        return payload
    finally:
        os.close(descriptor)


def _stable_regular_file_sha256(path: str | Path, *, label: str) -> tuple[str, int]:
    candidate = Path(path).expanduser()
    lexical = Path(
        os.path.abspath(
            os.fspath(candidate if candidate.is_absolute() else Path.cwd() / candidate)
        )
    )
    before = os.lstat(lexical)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        _fail("UNSAFE_AUTHORITY_FILE", label)
    descriptor = os.open(lexical, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if _file_signature(opened) != _file_signature(before):
            _fail("AUTHORITY_CHANGED_DURING_OPEN", label)
        digest = hashlib.sha256()
        observed = 0
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                observed += len(chunk)
        after = os.lstat(lexical)
        if (
            _file_signature(after) != _file_signature(before)
            or observed != before.st_size
        ):
            _fail("AUTHORITY_CHANGED_DURING_READ", label)
        return digest.hexdigest(), observed
    finally:
        os.close(descriptor)


def _json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise FundamentalSuccessorMaintenanceError(
            "AUTHORITY_JSON_INVALID", label
        ) from exc
    if not isinstance(value, dict):
        _fail("AUTHORITY_JSON_INVALID", label)
    return value


def _resolve_workspace_path(value: str | Path, *, workspace_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = workspace_root / path
    return path.resolve(strict=True)


def _resolve_pointer_reference(
    value: str,
    *,
    workspace_root: Path,
    pointer_parent: Path,
) -> Path:
    path = Path(str(value or "")).expanduser()
    if path.is_absolute():
        return path.resolve(strict=True)
    candidates = (workspace_root / path, pointer_parent / path)
    for candidate in candidates:
        try:
            return candidate.resolve(strict=True)
        except FileNotFoundError:
            continue
    _fail("POINTER_REFERENCE_MISSING", str(value))
    raise AssertionError("unreachable")


def _exact_date(value: Any, *, label: str) -> str:
    text = "".join(character for character in str(value or "") if character.isdigit())
    if len(text) != 8:
        _fail("DATE_INVALID", label)
    try:
        datetime.strptime(text, "%Y%m%d")
    except ValueError as exc:
        raise FundamentalSuccessorMaintenanceError("DATE_INVALID", label) from exc
    return text


def _valid_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _parent_cutoff(pointer: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    provenance = dict(manifest.get("primary_provenance", {}) or {})
    if provenance.get("schema_version") == "cn-fundamental-primary-provenance.v3":
        boundary = dict(
            dict(provenance.get("successor_chain", {}) or {}).get(
                "append_boundary", {}
            )
            or {}
        )
        return _exact_date(boundary.get("target_cutoff"), label="v3 parent cutoff")
    provider = dict(
        dict(manifest.get("metadata", {}) or {}).get("provider_manifest", {}) or {}
    )
    cutoff = provider.get("strict_pit_as_of")
    if cutoff is None:
        cutoff = dict(manifest.get("metadata", {}) or {}).get("as_of")
    resolved = _exact_date(cutoff, label="v2 parent cutoff")
    if dict(pointer.get("metadata", {}) or {}).get("gate2_passed") is not True:
        _fail("PREDECESSOR_GATE_NOT_PASSED")
    return resolved


def _financial_support_start(manifest: Mapping[str, Any], *, parent_cutoff: str) -> str:
    provider = dict(
        dict(manifest.get("metadata", {}) or {}).get("provider_manifest", {}) or {}
    )
    value = provider.get("financial_start_date")
    if value:
        return _exact_date(value, label="financial support start")
    provenance = dict(manifest.get("primary_provenance", {}) or {})
    refs = dict(provenance.get("permanent_support_refs", {}) or {})
    for reference in refs.values():
        if not isinstance(reference, Mapping):
            continue
        candidate = reference.get("support_start")
        if candidate:
            return _exact_date(candidate, label="successor support start")
    # A later v3 predecessor keeps the original full-support scan contract.
    chain = dict(provenance.get("successor_chain", {}) or {})
    root_cutoff = dict(chain.get("original_seam", {}) or {}).get("cutoff")
    if root_cutoff:
        root = _exact_date(root_cutoff, label="original successor seam")
        if root <= parent_cutoff:
            return "20190806"
    _fail("SUPPORT_START_UNAVAILABLE")
    raise AssertionError("unreachable")


def _parent_authority(
    canonical_root: Path,
    *,
    expected_pointer_sha256: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    bytes,
    bytes,
    dict[str, Path],
    dict[str, Any],
]:
    pointer_path = canonical_root / FUNDAMENTAL_POINTER_FILENAME
    pointer_bytes = _stable_regular_bytes(pointer_path, label="Fundamental pointer")
    observed_pointer_sha256 = _sha256(pointer_bytes)
    if observed_pointer_sha256 != str(expected_pointer_sha256).strip().lower():
        _fail("PREDECESSOR_POINTER_CAS_MISMATCH")
    raw_pointer = _json_object(pointer_bytes, label="Fundamental pointer")
    validated = load_fundamental_pointer(canonical_root)
    if validated is None or validated.get("primary_provenance_verified") is not True:
        _fail("PREDECESSOR_PRIMARY_PROVENANCE_INVALID")
    manifest_path = _resolve_pointer_reference(
        str(raw_pointer.get("manifest_path") or ""),
        workspace_root=canonical_root.parents[2],
        pointer_parent=canonical_root,
    )
    manifest_bytes = _stable_regular_bytes(manifest_path, label="Fundamental manifest")
    raw_manifest = _json_object(manifest_bytes, label="Fundamental manifest")
    if raw_manifest != dict(validated.get("manifest", {}) or {}):
        _fail("PREDECESSOR_MANIFEST_READBACK_MISMATCH")
    cutoff = _parent_cutoff(raw_pointer, raw_manifest)
    closure = capture_parent_closure(
        raw_pointer,
        raw_manifest,
        cutoff=cutoff,
        generation_root=canonical_root,
        pointer_bytes=pointer_bytes,
        manifest_bytes=manifest_bytes,
    )
    pointer_tables = dict(raw_pointer.get("tables", {}) or {})
    parent_tables = {
        name: _resolve_pointer_reference(
            str(pointer_tables.get(name) or ""),
            workspace_root=canonical_root.parents[2],
            pointer_parent=canonical_root,
        )
        for name in FUNDAMENTAL_TABLES
    }
    return (
        raw_pointer,
        raw_manifest,
        pointer_bytes,
        manifest_bytes,
        parent_tables,
        closure,
    )


def _scope_symbols(scope_payload: Mapping[str, Any]) -> list[str]:
    values = scope_payload.get("full_a")
    if not isinstance(values, list):
        _fail("FULL_A_SCOPE_MISSING")
    symbols = sorted({_canonical_symbol(value, label="target scope") for value in values})
    if not symbols or len(symbols) != len(values):
        _fail("FULL_A_SCOPE_INVALID")
    return symbols


def _parent_subjects(parent_tables: Mapping[str, Path]) -> tuple[set[str], dict[str, Any]]:
    union: set[str] = set()
    per_table: dict[str, Any] = {}
    for name in FUNDAMENTAL_TABLES:
        path = parent_tables[name]
        schema = pq.read_schema(path)
        if "ts_code" not in schema.names:
            subjects: set[str] = set()
        else:
            values = pq.read_table(path, columns=["ts_code"]).column("ts_code").to_pylist()
            subjects = {
                _canonical_symbol(value, label=f"parent {name}")
                for value in values
            }
        union.update(subjects)
        ordered = sorted(subjects)
        per_table[name] = {
            "subject_count": len(ordered),
            "subject_keyset_sha256": _sha256(_canonical_json_bytes(ordered)),
        }
    if not union:
        _fail("PARENT_SUBJECT_SCOPE_EMPTY")
    return union, per_table


def _pit_expected_subjects(
    membership_path: Path,
    *,
    sessions: Sequence[str],
) -> dict[str, set[str]]:
    required = ["symbol", "effective_from", "effective_to"]
    schema = pq.read_schema(membership_path)
    if not set(required).issubset(schema.names):
        _fail("PIT_MEMBERSHIP_SCHEMA_INVALID")
    rows = pq.read_table(membership_path, columns=required).to_pylist()
    result = {session: set() for session in sessions}
    for row in rows:
        effective_from = _exact_date(row.get("effective_from"), label="PIT effective_from")
        effective_to_raw = str(row.get("effective_to") or "")
        effective_to = (
            _exact_date(effective_to_raw, label="PIT effective_to")
            if effective_to_raw
            else ""
        )
        if effective_to and effective_to < effective_from:
            _fail("PIT_MEMBERSHIP_INTERVAL_INVALID", str(row.get("symbol") or ""))
        active_sessions = [
            session
            for session in sessions
            if effective_from <= session and (not effective_to or session <= effective_to)
        ]
        if not active_sessions:
            continue
        symbol = _canonical_symbol(row.get("symbol"), label="PIT membership")
        for session in active_sessions:
            result[session].add(symbol)
    if any(not result[session] for session in sessions):
        _fail("PIT_EXPECTED_SCOPE_EMPTY")
    return result


def _subject_scope_closure(
    *,
    parent_subjects: set[str],
    parent_evidence: Mapping[str, Any],
    pit_expected_by_session: Mapping[str, set[str]],
    observed_by_session: Mapping[str, set[str]],
    target_scope: set[str],
    target: str,
    authority_sha256: Mapping[str, str],
) -> tuple[list[str], dict[str, Any]]:
    sessions = sorted(pit_expected_by_session)
    if sessions != sorted(observed_by_session):
        _fail("SUBJECT_SCOPE_SESSION_SET_MISMATCH")
    union = set(parent_subjects) | set(target_scope)
    for session in sessions:
        union.update(pit_expected_by_session[session])
        union.update(observed_by_session[session])
    symbols = sorted(union)
    body = {
        "alias_transformations": [],
        "authority_sha256": dict(authority_sha256),
        "frozen_before_provider_capture": True,
        "identity_policy": "EXACT_CANONICAL_TS_CODE_NO_ALIASES",
        "parent_prefix": {
            "subject_count": len(parent_subjects),
            "subject_keyset_sha256": _sha256(
                _canonical_json_bytes(sorted(parent_subjects))
            ),
            "tables": dict(parent_evidence),
        },
        "per_session": {
            session: {
                "observed_bar_subject_count": len(observed_by_session[session]),
                "observed_bar_subject_keyset_sha256": _sha256(
                    _canonical_json_bytes(sorted(observed_by_session[session]))
                ),
                "pit_expected_subject_count": len(pit_expected_by_session[session]),
                "pit_expected_subject_keyset_sha256": _sha256(
                    _canonical_json_bytes(sorted(pit_expected_by_session[session]))
                ),
            }
            for session in sessions
        },
        "projection_policy": (
            "PARENT_PREFIX_UNION_DELTA_PIT_EXPECTED_UNION_"
            "DELTA_OBSERVED_BARS_UNION_TARGET_FULL_A"
        ),
        "schema_version": "cn-fundamental-successor-subject-scope-closure.v1",
        "status": "closed",
        "subject_count": len(symbols),
        "subject_keyset_sha256": _sha256(_canonical_json_bytes(symbols)),
        "subject_symbols": symbols,
        "target_cutoff": target,
        "target_scope_count": len(target_scope),
        "target_scope_keyset_sha256": _sha256(
            _canonical_json_bytes(sorted(target_scope))
        ),
    }
    body["closure_sha256"] = canonical_json_sha256(body)
    return symbols, body


def _bar_keys(
    table_root: Path,
    *,
    sessions: Sequence[str],
    scope: set[str] | None = None,
) -> set[tuple[str, str]]:
    by_partition = sorted({(date[:4], date[4:6]) for date in sessions})
    wanted = set(sessions)
    output: set[tuple[str, str]] = set()
    for year, month in by_partition:
        part = table_root / f"year={year}" / f"month={month}" / "part.parquet"
        payload = pq.read_table(part, columns=["ts_code", "trade_date"]).to_pydict()
        symbols = payload.get("ts_code", [])
        dates = payload.get("trade_date", [])
        for symbol_value, date_value in zip(symbols, dates):
            symbol = _canonical_symbol(symbol_value, label="canonical bar")
            trade_date = str(date_value or "").strip()
            if (scope is None or symbol in scope) and trade_date in wanted:
                key = (symbol, trade_date)
                if key in output:
                    _fail("DUPLICATE_CANONICAL_BAR_KEY", f"{symbol}|{trade_date}")
                output.add(key)
    observed_sessions = {trade_date for _symbol, trade_date in output}
    if observed_sessions != wanted:
        _fail("CANONICAL_BAR_SESSION_SET_MISMATCH")
    return output


def _history_keyset(
    history: Mapping[str, Any],
    *,
    parent_cutoff: str,
    target_cutoff: str,
    scope_symbols: Sequence[str],
    coverage_scope_symbols: Sequence[str],
    observed_bar_keys: set[tuple[str, str]],
    market_inactive_symbols: Sequence[str],
) -> tuple[dict[str, Any], list[str], dict[str, int], dict[str, int]]:
    if (
        history.get("schema_version") != EXPECTED_HISTORY_AUDIT_SCHEMA
        or history.get("history_audit_status") != "passed"
        or history.get("full_window_recomputed") is not True
        or history.get("prior_trade_dates_reused") != 0
        or history.get("synthetic_bar_count") != 0
        or list(history.get("history_unresolved_gap_dates", []) or [])
    ):
        _fail("HISTORY_AUDIT_NOT_CLOSED")
    for field in ("target_trade_date", "effective_trade_date", "stable_trade_date"):
        if _exact_date(history.get(field), label=f"history {field}") != target_cutoff:
            _fail("HISTORY_AUDIT_TARGET_MISMATCH", field)
    per_date = {
        str(row.get("trade_date") or ""): dict(row)
        for row in list(history.get("per_date", []) or [])
        if isinstance(row, Mapping)
    }
    sessions = sorted(
        date for date in per_date if parent_cutoff < date <= target_cutoff
    )
    if not sessions or sessions[-1] != target_cutoff:
        _fail("HISTORY_AUDIT_SUCCESSOR_WINDOW_MISSING")
    scope = set(scope_symbols)
    inactive_authority = set(market_inactive_symbols).intersection(scope)
    observed: set[tuple[str, str]] = set()
    suspended: set[tuple[str, str]] = set()
    inactive: set[tuple[str, str]] = set()
    delisted: set[tuple[str, str]] = set()
    prelisting: set[tuple[str, str]] = set()
    true_missing: set[tuple[str, str]] = set()
    for trade_date in sessions:
        row = per_date[trade_date]
        if (
            row.get("status") != "passed"
            or list(row.get("blockers", []) or [])
            or row.get("classification_sets_disjoint") is not True
            or row.get("classification_union_complete") is not True
            or list(row.get("unknown_membership_symbols", []) or [])
        ):
            _fail("HISTORY_DATE_CLASSIFICATION_INVALID", trade_date)
        date_observed = {
            symbol for symbol, date_value in observed_bar_keys if date_value == trade_date
        }
        date_suspended = set().union(
            *(
                set(row.get(field, []) or [])
                for field in (
                    "verified_exact_suspended_absent",
                    "verified_suspended_absent",
                    "verified_suspension_continuity_absent",
                    "verified_nontrading_bak_daily_zero",
                )
            )
        ).intersection(scope)
        date_prelisting = set(row.get("excluded_prelisting_symbols", []) or []).intersection(scope)
        audit_delisted = (
            set(row.get("excluded_delisted_symbols", []) or [])
            | set(row.get("excluded_delisted_on_target_symbols", []) or [])
        ).intersection(scope)
        date_delisted = audit_delisted.difference(inactive_authority)
        audit_inactive = set(
            row.get("verified_inactive_or_prelisting_absent", []) or []
        ).intersection(scope)
        date_inactive = (
            audit_inactive.union(inactive_authority)
            .difference(date_prelisting)
            .difference(date_delisted)
        )
        date_missing = set(row.get("true_missing_symbols", []) or []).intersection(scope)
        categories = (
            date_observed,
            date_suspended,
            date_inactive,
            date_delisted,
            date_prelisting,
            date_missing,
        )
        seen: set[str] = set()
        for category in categories:
            if seen.intersection(category):
                _fail("SUCCESSOR_CLASSIFICATION_OVERLAP", trade_date)
            seen.update(category)
        if seen != scope or date_missing:
            _fail("SUCCESSOR_CLASSIFICATION_NOT_CLOSED", trade_date)
        observed.update((symbol, trade_date) for symbol in date_observed)
        suspended.update((symbol, trade_date) for symbol in date_suspended)
        inactive.update((symbol, trade_date) for symbol in date_inactive)
        delisted.update((symbol, trade_date) for symbol in date_delisted)
        prelisting.update((symbol, trade_date) for symbol in date_prelisting)
        true_missing.update((symbol, trade_date) for symbol in date_missing)
    expected = {(symbol, trade_date) for symbol in scope for trade_date in sessions}
    closure = build_keyset_closure(
        observed_bar_keys=observed,
        daily_basic_keys=observed,
        suspended_keys=suspended,
        inactive_keys=inactive,
        delisted_keys=delisted,
        prelisting_keys=prelisting,
        true_missing_keys=true_missing,
        expected_scope_keys=expected,
    )
    target_counts = {
        "observed": sum(date == target_cutoff for _symbol, date in observed),
        "suspended": sum(date == target_cutoff for _symbol, date in suspended),
        "inactive": sum(date == target_cutoff for _symbol, date in inactive),
        "delisted": sum(date == target_cutoff for _symbol, date in delisted),
        "prelisting": sum(date == target_cutoff for _symbol, date in prelisting),
        "true_missing": sum(date == target_cutoff for _symbol, date in true_missing),
    }
    coverage_scope = set(coverage_scope_symbols)
    coverage_counts = {
        "observed": sum(
            date == target_cutoff and symbol in coverage_scope
            for symbol, date in observed
        ),
        "suspended": sum(
            date == target_cutoff and symbol in coverage_scope
            for symbol, date in suspended
        ),
        "inactive": sum(
            date == target_cutoff and symbol in coverage_scope
            for symbol, date in inactive
        ),
        "delisted": sum(
            date == target_cutoff and symbol in coverage_scope
            for symbol, date in delisted
        ),
        "prelisting": sum(
            date == target_cutoff and symbol in coverage_scope
            for symbol, date in prelisting
        ),
        "true_missing": sum(
            date == target_cutoff and symbol in coverage_scope
            for symbol, date in true_missing
        ),
    }
    return closure, sessions, target_counts, coverage_counts


def _immutable_ref(path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256(payload),
        "byte_length": len(payload),
    }


@contextmanager
def _configured_token_for_official_client() -> Iterator[None]:
    previous = os.environ.get("TUSHARE_TOKEN")
    installed = False
    if not previous:
        from quant_investor.config import config

        token = str(config.TUSHARE_TOKEN or "").strip()
        if not token:
            _fail("TUSHARE_TOKEN_MISSING")
        os.environ["TUSHARE_TOKEN"] = token
        installed = True
    try:
        yield
    finally:
        if installed:
            os.environ.pop("TUSHARE_TOKEN", None)


def _private_fileset_root(path: str | Path, *, workspace_root: Path) -> Path:
    root = Path(path).expanduser()
    if not root.is_absolute():
        root = workspace_root / root
    root = root.absolute()
    root.parent.mkdir(parents=True, exist_ok=True)
    if not root.exists():
        root.mkdir(mode=0o700)
    metadata = os.lstat(root)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        _fail("UNSAFE_SUPPORT_FILESET_ROOT")
    os.chmod(root, 0o700)
    return root


def _relative_support_refs(
    evidence_sha256: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    wanted = {
        "predecessor_pointer": "authority/predecessor_pointer.json",
        "predecessor_manifest": "authority/predecessor_manifest.json",
        # This sealed source manifest transitively binds the source binding,
        # every request/table file, all three exact pointer byte strings, and
        # their immutable-reference closure.
        "support_manifest": "source/provider_manifest.json",
    }
    refs: dict[str, dict[str, Any]] = {}
    for name, relative in wanted.items():
        digest = evidence_sha256.get(relative)
        if digest is None:
            _fail("PERMANENT_SUPPORT_EVIDENCE_MISSING", relative)
        refs[name] = {
            "path": relative,
            "sha256": digest,
        }
    return refs


def _physical_memory_bytes() -> int:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        page_count = int(os.sysconf("SC_PHYS_PAGES"))
    except (OSError, TypeError, ValueError):
        _fail("PHYSICAL_MEMORY_PREFLIGHT_FAILED")
    total = page_size * page_count
    if page_size < 1 or page_count < 1 or total < 1024 * 1024 * 1024:
        _fail("PHYSICAL_MEMORY_PREFLIGHT_FAILED")
    return total


def _available_memory_bytes() -> int:
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
                if line.startswith("MemAvailable:"):
                    value = int(line.split()[1]) * 1024
                    if value > 0:
                        return value
        except (OSError, UnicodeError, ValueError, IndexError):
            pass
    if sys.platform == "darwin":
        try:
            completed = subprocess.run(
                ["/usr/bin/vm_stat"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            first, *lines = completed.stdout.splitlines()
            match = re.search(r"page size of ([0-9]+) bytes", first)
            if match is None:
                _fail("AVAILABLE_MEMORY_PREFLIGHT_FAILED")
            counts: dict[str, int] = {}
            for line in lines:
                if ":" not in line:
                    continue
                name, raw = line.split(":", 1)
                value = raw.strip().rstrip(".")
                if value.isdigit():
                    counts[name] = int(value)
            pages = sum(
                counts.get(name, 0)
                for name in (
                    "Pages free",
                    "Pages inactive",
                    "Pages speculative",
                    "Pages purgeable",
                )
            )
            available = pages * int(match.group(1))
            if available > 0:
                return available
        except FundamentalSuccessorMaintenanceError:
            raise
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    _fail("AVAILABLE_MEMORY_PREFLIGHT_FAILED")


def _resident_memory_bytes() -> int:
    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (OSError, ValueError):
        _fail("RESIDENT_MEMORY_PREFLIGHT_FAILED")
    return value if sys.platform == "darwin" else value * 1024


def _rlimit_headroom_bytes(*, resident_memory_bytes: int) -> int:
    candidates: list[int] = []
    for name in ("RLIMIT_AS", "RLIMIT_DATA"):
        identifier = getattr(resource, name, None)
        if identifier is None:
            continue
        try:
            soft, _hard = resource.getrlimit(identifier)
        except (OSError, ValueError):
            _fail("MEMORY_RLIMIT_PREFLIGHT_FAILED")
        if soft not in {resource.RLIM_INFINITY, -1}:
            candidates.append(max(0, int(soft) - resident_memory_bytes))
    return min(candidates) if candidates else 2**63 - 1


def _existing_ancestor(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    if not current.exists():
        _fail("RESOURCE_PATH_UNAVAILABLE", str(path))
    return current


def _resource_preflight_receipt(
    *,
    phase: str,
    source_root: Path,
    staging_root: Path,
    canonical_root: Path,
    parent_storage_bytes: int,
    planned_request_count: int,
    source_storage_bytes: int | None = None,
    source_table_bytes: int | None = None,
    predecessor_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    if (
        phase not in {"PRE_PROVIDER", "POST_CAPTURE"}
        or type(parent_storage_bytes) is not int
        or parent_storage_bytes < 1
        or type(planned_request_count) is not int
        or planned_request_count < 1
    ):
        _fail("RESOURCE_PREFLIGHT_INPUT_INVALID")
    if phase == "POST_CAPTURE" and (
        type(source_storage_bytes) is not int
        or source_storage_bytes < 1
        or type(source_table_bytes) is not int
        or source_table_bytes < 1
        or type(predecessor_receipt_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", predecessor_receipt_sha256) is None
    ):
        _fail("RESOURCE_PREFLIGHT_INPUT_INVALID")
    physical_memory = _physical_memory_bytes()
    available_memory = _available_memory_bytes()
    resident_memory = _resident_memory_bytes()
    rlimit_headroom = _rlimit_headroom_bytes(
        resident_memory_bytes=resident_memory
    )
    effective_memory = min(physical_memory, available_memory, rlimit_headroom)
    if phase == "PRE_PROVIDER":
        source_estimate = max(
            1024 * 1024 * 1024,
            planned_request_count * 128 * 1024,
        )
        suffix_estimate = planned_request_count * 64 * 1024
    else:
        assert source_storage_bytes is not None
        assert source_table_bytes is not None
        source_estimate = source_storage_bytes
        suffix_estimate = source_table_bytes * 4
    candidate_estimate = parent_storage_bytes + suffix_estimate
    generation_with_evidence = candidate_estimate + source_estimate
    category_values = {
        "canonical_final": generation_with_evidence,
        "canonical_orphan": generation_with_evidence,
        "canonical_temp": generation_with_evidence,
        "fsync_reserve": 64 * 1024 * 1024,
        "rollback_reserve": parent_storage_bytes,
        "source_capture": source_estimate,
        "staging_temp": generation_with_evidence,
    }
    category_locations = {
        "source_capture": source_root,
        "staging_temp": staging_root,
        "canonical_temp": canonical_root,
        "canonical_final": canonical_root,
        "canonical_orphan": canonical_root,
        "rollback_reserve": canonical_root,
        "fsync_reserve": canonical_root,
    }
    devices: dict[str, dict[str, Any]] = {}
    categories: dict[str, dict[str, Any]] = {}
    for name, required_bytes in category_values.items():
        path = category_locations[name]
        ancestor = _existing_ancestor(path)
        device = str(os.stat(ancestor).st_dev)
        free = int(shutil.disk_usage(ancestor).free)
        categories[name] = {
            "device": device,
            "path": str(path),
            "required_bytes": required_bytes,
        }
        row = devices.setdefault(
            device,
            {
                "free_bytes": free,
                "required_before_margin_bytes": 0,
                "roles": [],
            },
        )
        row["free_bytes"] = min(int(row["free_bytes"]), free)
        row["required_before_margin_bytes"] += required_bytes
        row["roles"].append(name)
    blockers: list[str] = []
    for device, row in devices.items():
        subtotal = int(row["required_before_margin_bytes"])
        margin = (subtotal + 3) // 4
        required = subtotal + margin
        row["margin_25_percent_bytes"] = margin
        row["required_with_margin_bytes"] = required
        row["roles"] = sorted(row["roles"])
        if int(row["free_bytes"]) < required:
            blockers.append(f"INSUFFICIENT_DISK_DEVICE_{device}")
    source_device = categories["source_capture"]["device"]
    source_device_row = devices[source_device]
    non_source_reserve = (
        int(source_device_row["required_with_margin_bytes"])
        - int(categories["source_capture"]["required_bytes"])
    )
    table_memory_limit = effective_memory // 2
    maximum_record_bytes = min(32 * 1024 * 1024, table_memory_limit // 4)
    if effective_memory < 128 * 1024 * 1024:
        blockers.append("INSUFFICIENT_AVAILABLE_MEMORY_OR_RLIMIT")
    body: dict[str, Any] = {
        "blockers": blockers,
        "categories": categories,
        "devices": devices,
        "maximum_record_bytes": maximum_record_bytes,
        "minimum_source_free_disk_bytes": max(
            256 * 1024 * 1024,
            non_source_reserve,
        ),
        "parent_storage_bytes": parent_storage_bytes,
        "phase": phase,
        "physical_memory_bytes": physical_memory,
        "available_memory_bytes": available_memory,
        "resident_memory_bytes": resident_memory,
        "rlimit_headroom_bytes": rlimit_headroom,
        "effective_memory_headroom_bytes": effective_memory,
        "stream_batch_rows": 2_048,
        "stream_batch_bytes": 16 * 1024 * 1024,
        "planned_request_count": planned_request_count,
        "predecessor_receipt_sha256": predecessor_receipt_sha256,
        "schema_version": SAFE_SUCCESSOR_RESOURCE_PREFLIGHT_SCHEMA,
        "source_storage_bytes": source_storage_bytes,
        "source_table_bytes": source_table_bytes,
        "status": "PASS" if not blockers else "BLOCKED",
        "table_memory_limit_bytes": table_memory_limit,
    }
    body["receipt_sha256"] = canonical_json_sha256(body)
    if blockers:
        _fail("RESOURCE_PREFLIGHT_BLOCKED", ",".join(blockers))
    return body


def _derivation_resource_budget(
    *,
    post_capture_receipt: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    keyset: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        post_capture_receipt.get("phase") != "POST_CAPTURE"
        or post_capture_receipt.get("status") != "PASS"
    ):
        _fail("DERIVATION_RESOURCE_BUDGET_INPUT_INVALID")
    table_files = dict(source_manifest.get("table_files", {}) or {})
    required_tables = {
        "balancesheet",
        "cashflow",
        "daily_basic",
        "fina_indicator",
        "forecast",
        "income",
    }
    if set(table_files) != required_tables:
        _fail("DERIVATION_RESOURCE_BUDGET_INPUT_INVALID")
    row_counts: dict[str, int] = {}
    for table, reference in table_files.items():
        if not isinstance(reference, Mapping):
            _fail("DERIVATION_RESOURCE_BUDGET_INPUT_INVALID")
        row_count = reference.get("row_count")
        if type(row_count) is not int or row_count < 0:
            _fail("DERIVATION_RESOURCE_BUDGET_INPUT_INVALID")
        row_counts[table] = row_count
    daily_keys = list(keyset.get("daily_basic_keys", []) or [])
    expected_keys = list(keyset.get("expected_scope_keys", []) or [])
    symbols = {
        str(value).split("|", 1)[0]
        for value in expected_keys
        if type(value) is str and "|" in value
    }
    if not symbols or len(daily_keys) != row_counts["daily_basic"]:
        _fail("DERIVATION_RESOURCE_BUDGET_INPUT_INVALID")
    effective = post_capture_receipt.get("effective_memory_headroom_bytes")
    source_table_bytes = post_capture_receipt.get("source_table_bytes")
    if (
        type(effective) is not int
        or effective < 128 * 1024 * 1024
        or type(source_table_bytes) is not int
        or source_table_bytes < 1
    ):
        _fail("DERIVATION_RESOURCE_BUDGET_INPUT_INVALID")
    body: dict[str, Any] = {
        "schema_version": (
            "cn-fundamental-successor-derivation-accumulator-budget.v1"
        ),
        "status": "PASS",
        "period_anchor_row_limit": len(symbols),
        "forecast_anchor_row_limit": len(symbols),
        "period_delta_row_limit": sum(
            row_counts[table]
            for table in ("fina_indicator", "income", "balancesheet", "cashflow")
        ),
        "period_lineage_row_limit": sum(
            row_counts[table]
            for table in ("fina_indicator", "income", "balancesheet", "cashflow")
        ),
        "forecast_delta_row_limit": row_counts["forecast"],
        "daily_basic_row_limit": len(daily_keys),
        "total_accumulator_byte_limit": min(
            effective // 4,
            max(64 * 1024 * 1024, source_table_bytes * 8),
        ),
        "effective_memory_headroom_bytes": effective,
        "post_capture_receipt_sha256": post_capture_receipt["receipt_sha256"],
    }
    body["binding_sha256"] = canonical_json_sha256(body)
    return body


def _financial_support_closure(
    *,
    source_plan: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    target_cutoff: str,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    requests = list(source_plan.get("requests", []) or [])
    receipts = list(source_manifest.get("request_receipts", []) or [])
    if len(requests) != len(receipts):
        _fail("INCOME_SUPPORT_RECEIPT_SET_MISMATCH")
    captured: list[dict[str, str]] = []
    absences: list[dict[str, Any]] = []
    requested: set[tuple[str, str, str]] = set()
    closed: set[tuple[str, str, str]] = set()
    for request, receipt in zip(requests, receipts, strict=True):
        if request.get("partition_type") != "EXACT_SYMBOL_REPORT_PERIOD_SUPPORT":
            continue
        params = dict(request.get("params", {}) or {})
        key = (
            str(request.get("table") or ""),
            str(params.get("ts_code") or ""),
            str(params.get("period") or ""),
        )
        requested.add(key)
        accepted = receipt.get("accepted_count")
        status = receipt.get("status")
        if type(accepted) is not int or accepted < 0:
            _fail("INCOME_SUPPORT_RECEIPT_INVALID")
        if accepted > 0 and status == "AVAILABLE":
            captured.append(
                {"table": key[0], "ts_code": key[1], "end_date": key[2]}
            )
        elif accepted == 0 and status == "EMPTY":
            absences.append(
                {
                    "status": "PROVEN_ABSENT",
                    "table": key[0],
                    "symbol": key[1],
                    "end_date": key[2],
                    "available_through": target_cutoff,
                    "evidence_sha256": str(receipt.get("receipt_sha256") or ""),
                }
            )
        else:
            _fail("INCOME_SUPPORT_RECEIPT_INVALID")
        closed.add(key)
    if closed != requested:
        _fail("INCOME_SUPPORT_RECEIPT_SET_MISMATCH")
    captured.sort(key=lambda row: (row["table"], row["ts_code"], row["end_date"]))
    absences.sort(key=lambda row: (row["table"], row["symbol"], row["end_date"]))
    return captured, absences


def _aggregate_implementation_sha256() -> str:
    digest = hashlib.sha256()
    package_root = Path(__file__).parents[1]
    for path in sorted(
        (
            Path(__file__),
            Path(__file__).with_name("fundamental_incremental.py"),
            Path(__file__).with_name("fundamental_historical_taint.py"),
            Path(__file__).with_name("fundamental_successor_source.py"),
            Path(__file__).with_name("tushare_transport.py"),
        ),
        key=lambda value: value.relative_to(package_root).as_posix(),
    ):
        relative = path.relative_to(package_root).as_posix()
        payload = _stable_regular_bytes(path, label=f"implementation {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(payload)
    return digest.hexdigest()


def run_cn_fundamental_safe_successor(
    *,
    as_of: str,
    run_id: str,
    staging_root: str | Path,
    canonical_root: str | Path,
    expected_pointer_sha256: str,
    canonical_market_pointer_path: str | Path,
    canonical_pit_pointer_path: str | Path,
    canonical_membership_path: str | Path,
    canonical_scope_path: str | Path,
    history_audit_path: str | Path,
    expected_history_audit_sha256: str,
    support_fileset_root: str | Path,
    allow_live: bool,
    universes: Sequence[str] | str,
    max_attempts: int = 3,
    retry_backoff_seconds: Sequence[float] = (0.5, 1.0),
    requests_per_second: float = 8.0,
    client: Any | None = None,
    taint_analysis_dry_run: bool = False,
    audit_run_root: str | Path | None = None,
    append_first: bool = False,
    historical_taint_evidence: Sequence[Mapping[str, Any]] = (),
    income_support_dependencies: Sequence[Mapping[str, Any]] = (),
    financial_support_dependencies: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build a successor, or run a promotion-ineligible taint diagnostic."""

    if not allow_live:
        _fail("SAFE_SUCCESSOR_REQUIRES_ALLOW_LIVE")
    universe_list = (
        [item.strip().lower() for item in universes.split(",") if item.strip()]
        if isinstance(universes, str)
        else [str(item).strip().lower() for item in universes]
    )
    if universe_list != ["full_a"]:
        _fail("SAFE_SUCCESSOR_REQUIRES_FULL_A")
    if append_first and taint_analysis_dry_run:
        _fail("APPEND_FIRST_TAINT_DRY_RUN_CONFLICT")
    if append_first and not historical_taint_evidence:
        _fail("APPEND_FIRST_HISTORICAL_TAINT_EVIDENCE_REQUIRED")
    if not append_first and historical_taint_evidence:
        _fail("HISTORICAL_TAINT_EVIDENCE_WITHOUT_APPEND_FIRST")
    if (income_support_dependencies or financial_support_dependencies) and not append_first:
        _fail("FINANCIAL_SUPPORT_DEPENDENCY_WITHOUT_APPEND_FIRST")
    target = _exact_date(as_of, label="target cutoff")
    generation_id = str(run_id or "").strip()
    if not generation_id:
        _fail("SAFE_SUCCESSOR_RUN_ID_REQUIRED")
    workspace_root = Path.cwd().resolve(strict=True)
    canonical = _resolve_workspace_path(canonical_root, workspace_root=workspace_root)
    audit_root: Path | None = None
    if taint_analysis_dry_run:
        if audit_run_root is None:
            _fail("TAINT_AUDIT_ROOT_REQUIRED")
        audit_root = _private_audit_root(
            audit_run_root,
            workspace_root=workspace_root,
        )
        staging = audit_root / "_staging_forbidden"
        expected_capture_root = audit_root / "capture"
        supplied_capture_root = Path(support_fileset_root).expanduser().absolute()
        if supplied_capture_root != expected_capture_root:
            _fail("TAINT_CAPTURE_ROOT_OUTSIDE_AUDIT_ROOT")
    else:
        if audit_run_root is not None:
            _fail("TAINT_AUDIT_ROOT_WITHOUT_DRY_RUN")
        staging = Path(staging_root).expanduser()
        if not staging.is_absolute():
            staging = workspace_root / staging
        staging = staging.absolute()
        if staging.exists():
            _fail("STAGING_ROOT_EXISTS")
    (
        parent_pointer,
        parent_manifest,
        parent_pointer_bytes,
        parent_manifest_bytes,
        parent_tables,
        parent_closure,
    ) = _parent_authority(
        canonical,
        expected_pointer_sha256=expected_pointer_sha256,
    )
    parent_cutoff = str(parent_closure["cutoff"])
    if target <= parent_cutoff:
        _fail("TARGET_NOT_AFTER_PREDECESSOR")
    parent_subjects, parent_subject_evidence = _parent_subjects(parent_tables)

    market_pointer_path = _resolve_workspace_path(
        canonical_market_pointer_path,
        workspace_root=workspace_root,
    )
    market_pointer_bytes = _stable_regular_bytes(
        market_pointer_path,
        label="market pointer",
    )
    market_pointer = _json_object(market_pointer_bytes, label="market pointer")
    if (
        market_pointer.get("status") != "OK"
        or _exact_date(
            market_pointer.get("latest_complete_trade_date"),
            label="market latest complete",
        )
        != target
        or list(market_pointer.get("blockers", []) or [])
    ):
        _fail("MARKET_TARGET_NOT_READY")
    market_manifest_path = _resolve_pointer_reference(
        str(market_pointer.get("manifest_path") or ""),
        workspace_root=workspace_root,
        pointer_parent=market_pointer_path.parent,
    )
    market_manifest_bytes = _stable_regular_bytes(
        market_manifest_path,
        label="market snapshot manifest",
    )
    market_manifest = _json_object(
        market_manifest_bytes,
        label="market snapshot manifest",
    )
    if (
        market_manifest.get("snapshot_id") != market_pointer.get("snapshot_id")
        or market_manifest.get("latest_complete_trade_date") != target
        or market_manifest.get("readback_validated") is not True
    ):
        _fail("MARKET_MANIFEST_NOT_READY")

    pit_pointer_path = _resolve_workspace_path(
        canonical_pit_pointer_path,
        workspace_root=workspace_root,
    )
    pit_pointer_bytes = _stable_regular_bytes(pit_pointer_path, label="PIT pointer")
    pit_pointer = _json_object(pit_pointer_bytes, label="PIT pointer")
    membership_path = _resolve_workspace_path(
        canonical_membership_path,
        workspace_root=workspace_root,
    )
    membership_bytes = _stable_regular_bytes(membership_path, label="PIT membership")
    if (
        str(pit_pointer.get("canonical_path") or "") != str(membership_path)
        or str(pit_pointer.get("canonical_sha256") or "").lower()
        != _sha256(membership_bytes)
    ):
        _fail("PIT_MEMBERSHIP_BINDING_MISMATCH")
    pit_manifest_path = _resolve_workspace_path(
        str(pit_pointer.get("generation_manifest_path") or ""),
        workspace_root=workspace_root,
    )
    pit_manifest_bytes = _stable_regular_bytes(
        pit_manifest_path,
        label="PIT generation manifest",
    )
    if str(pit_pointer.get("generation_manifest_sha256") or "").lower() != _sha256(
        pit_manifest_bytes
    ):
        _fail("PIT_MANIFEST_BINDING_MISMATCH")

    scope_path = _resolve_workspace_path(
        canonical_scope_path,
        workspace_root=workspace_root,
    )
    scope_bytes = _stable_regular_bytes(scope_path, label="full-A scope")
    scope_payload = _json_object(scope_bytes, label="full-A scope")
    target_scope_symbols = _scope_symbols(scope_payload)
    coverage = dict(market_pointer.get("coverage", {}) or {})
    if (
        coverage.get("complete") is not True
        or int(coverage.get("expected_scope_count", -1)) != len(target_scope_symbols)
        or list(coverage.get("true_missing_symbols", []) or [])
        or coverage.get("classification_sets_disjoint") is not True
    ):
        _fail("MARKET_SCOPE_COVERAGE_NOT_CLOSED")

    history_path = _resolve_workspace_path(
        history_audit_path,
        workspace_root=workspace_root,
    )
    history_bytes = _stable_regular_bytes(history_path, label="history audit")
    if _sha256(history_bytes) != str(expected_history_audit_sha256).strip().lower():
        _fail("HISTORY_AUDIT_SHA_MISMATCH")
    history = _json_object(history_bytes, label="history audit")
    canonical_history = dict(history.get("canonical", {}) or {})
    if (
        canonical_history.get("latest_sha256") != _sha256(market_pointer_bytes)
        or canonical_history.get("snapshot_id") != market_pointer.get("snapshot_id")
    ):
        _fail("HISTORY_AUDIT_MARKET_BINDING_MISMATCH")
    sessions = sorted(
        date
        for date in list(history.get("audited_trade_dates", []) or [])
        if parent_cutoff < str(date) <= target
    )
    table_root = _resolve_pointer_reference(
        str(market_pointer.get("table_root") or ""),
        workspace_root=workspace_root,
        pointer_parent=market_pointer_path.parent,
    )
    observed_bar_keys = _bar_keys(
        table_root,
        sessions=sessions,
    )
    observed_by_session = {
        session: {
            symbol
            for symbol, trade_date in observed_bar_keys
            if trade_date == session
        }
        for session in sessions
    }
    pit_expected_by_session = _pit_expected_subjects(
        membership_path,
        sessions=sessions,
    )
    symbols, subject_scope_closure = _subject_scope_closure(
        parent_subjects=parent_subjects,
        parent_evidence=parent_subject_evidence,
        pit_expected_by_session=pit_expected_by_session,
        observed_by_session=observed_by_session,
        target_scope=set(target_scope_symbols),
        target=target,
        authority_sha256={
            "history_audit": _sha256(history_bytes),
            "market_manifest": _sha256(market_manifest_bytes),
            "market_pointer": _sha256(market_pointer_bytes),
            "pit_membership": _sha256(membership_bytes),
            "pit_pointer": _sha256(pit_pointer_bytes),
            "predecessor_manifest": _sha256(parent_manifest_bytes),
            "predecessor_pointer": _sha256(parent_pointer_bytes),
            "target_scope": _sha256(scope_bytes),
        },
    )
    subject_scope_bytes = _canonical_json_bytes(subject_scope_closure)
    keyset, audited_sessions, target_counts, coverage_target_counts = _history_keyset(
        history,
        parent_cutoff=parent_cutoff,
        target_cutoff=target,
        scope_symbols=symbols,
        coverage_scope_symbols=target_scope_symbols,
        observed_bar_keys=observed_bar_keys,
        market_inactive_symbols=list(coverage.get("inactive_symbols", []) or []),
    )
    if sessions != audited_sessions:
        _fail("SUCCESSOR_SESSION_SET_MISMATCH")
    if (
        coverage_target_counts["observed"]
        != int(coverage.get("observed_bar_count", -1))
        or coverage_target_counts["suspended"]
        != len(list(coverage.get("suspended_symbols", []) or []))
        or coverage_target_counts["inactive"]
        != len(list(coverage.get("inactive_symbols", []) or []))
        or coverage_target_counts["true_missing"] != 0
    ):
        _fail("TARGET_CLASSIFICATION_COUNT_MISMATCH")

    support_start = (
        (
            datetime.strptime(parent_cutoff, "%Y%m%d").date()
            + timedelta(days=1)
        ).strftime("%Y%m%d")
        if append_first
        else _financial_support_start(
            parent_manifest,
            parent_cutoff=parent_cutoff,
        )
    )
    source_plan = build_successor_support_plan(
        support_start=support_start,
        target_date=target,
        open_sessions=sessions,
        symbols=symbols,
        canonical_subject_scope_authority_sha256=subject_scope_closure[
            "closure_sha256"
        ],
        income_support_dependencies=income_support_dependencies,
        financial_support_dependencies=financial_support_dependencies,
    )
    implementation_sha256 = _aggregate_implementation_sha256()
    source_root = _private_fileset_root(
        support_fileset_root,
        workspace_root=workspace_root,
    )
    parent_storage_bytes = sum(path.stat().st_size for path in parent_tables.values())
    pre_provider_resource = _resource_preflight_receipt(
        phase="PRE_PROVIDER",
        source_root=source_root,
        staging_root=staging,
        canonical_root=canonical,
        parent_storage_bytes=parent_storage_bytes,
        planned_request_count=int(source_plan["planned_request_count"]),
    )
    immutable_refs = {
        "predecessor": {
            "generation_id": parent_pointer.get("generation_id"),
            "manifest_path": str(
                _resolve_pointer_reference(
                    str(parent_pointer.get("manifest_path") or ""),
                    workspace_root=workspace_root,
                    pointer_parent=canonical,
                )
            ),
            "manifest_sha256": _sha256(parent_manifest_bytes),
            "table_sha256": dict(parent_closure["table_sha256"]),
            "cutoff": parent_cutoff,
        },
        "market": {
            "snapshot_id": market_pointer.get("snapshot_id"),
            "manifest_path": str(market_manifest_path),
            "manifest_sha256": _sha256(market_manifest_bytes),
            "target": target,
        },
        "pit": {
            "generation_id": pit_pointer.get("generation_id"),
            "manifest_path": str(pit_manifest_path),
            "manifest_sha256": _sha256(pit_manifest_bytes),
            "membership_path": str(membership_path),
            "membership_sha256": _sha256(membership_bytes),
            "target": target,
        },
        "scope": {
            "path": str(scope_path),
            "sha256": _sha256(scope_bytes),
            "symbol_count": len(target_scope_symbols),
            "canonical_subject_count": len(symbols),
            "canonical_subject_scope_sha256": subject_scope_closure[
                "closure_sha256"
            ],
        },
        "history_audit": {
            "path": str(history_path),
            "sha256": _sha256(history_bytes),
            "target": target,
        },
        "live_pointer_paths": {
            "predecessor": str(canonical / FUNDAMENTAL_POINTER_FILENAME),
            "market": str(market_pointer_path),
            "pit": str(pit_pointer_path),
        },
        "resource_preflight": pre_provider_resource,
    }
    parent_manifest_path = _resolve_pointer_reference(
        str(parent_pointer.get("manifest_path") or ""),
        workspace_root=workspace_root,
        pointer_parent=canonical,
    )
    authority_freeze = {
        canonical / FUNDAMENTAL_POINTER_FILENAME: parent_pointer_bytes,
        parent_manifest_path: parent_manifest_bytes,
        market_pointer_path: market_pointer_bytes,
        market_manifest_path: market_manifest_bytes,
        pit_pointer_path: pit_pointer_bytes,
        pit_manifest_path: pit_manifest_bytes,
        membership_path: membership_bytes,
        scope_path: scope_bytes,
        history_path: history_bytes,
    }
    parent_table_freeze = {
        path: _stable_regular_file_sha256(
            path,
            label=f"parent table {name}",
        )
        for name, path in parent_tables.items()
    }
    if taint_analysis_dry_run:
        assert audit_root is not None
        frozen_state = _sealed(
            {
                "schema_version": "cn-fundamental-taint-frozen-state.v1",
                "run_id": generation_id,
                "parent_cutoff": parent_cutoff,
                "target_cutoff": target,
                "open_sessions": sessions,
                "dependency_contract_sha256": (
                    SUCCESSOR_FINANCIAL_DEPENDENCY_CONTRACT_SHA256
                ),
                "implementation_sha256": implementation_sha256,
                "authority_sha256": {
                    str(path): _sha256(payload)
                    for path, payload in authority_freeze.items()
                }
                | {
                    str(path): digest
                    for path, (digest, _size) in parent_table_freeze.items()
                },
                "live_provider_capture": True,
                "canonical_write": False,
                "staging_write": False,
                "promotion_authorized": False,
            },
            field="frozen_state_sha256",
        )
        _atomic_private_json(audit_root / "frozen_state.json", frozen_state)
    if client is None:
        from quant_investor.market.tushare_transport import (
            OfficialTushareHttpsClient,
        )

        client = OfficialTushareHttpsClient(
            timeout_seconds=30.0,
            strict_decimal_decode=True,
            max_response_items=20_000,
        )
    captured_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with _configured_token_for_official_client():
        source_manifest = acquire_successor_support(
            plan=source_plan,
            client=client,
            fileset_root=source_root,
            captured_pointer_bytes={
                "predecessor": parent_pointer_bytes,
                "market": market_pointer_bytes,
                "pit": pit_pointer_bytes,
            },
            immutable_refs=immutable_refs,
            implementation_sha256=implementation_sha256,
            captured_at=captured_at,
            max_attempts=int(max_attempts),
            retry_backoff_seconds=tuple(float(value) for value in retry_backoff_seconds),
            requests_per_second=float(requests_per_second),
            physical_memory_bytes=pre_provider_resource["physical_memory_bytes"],
            available_memory_bytes=pre_provider_resource[
                "available_memory_bytes"
            ],
            rlimit_headroom_bytes=pre_provider_resource[
                "rlimit_headroom_bytes"
            ],
            table_memory_limit_bytes=pre_provider_resource[
                "table_memory_limit_bytes"
            ],
            minimum_free_disk_bytes=pre_provider_resource[
                "minimum_source_free_disk_bytes"
            ],
            maximum_record_bytes=pre_provider_resource["maximum_record_bytes"],
        )
    source_validator = (
        validate_successor_capture_fileset
        if taint_analysis_dry_run
        else validate_successor_support_fileset
    )
    source_manifest = source_validator(
        source_root,
        expected_implementation_sha256=implementation_sha256,
    )
    if taint_analysis_dry_run:
        _verify_authority_freeze(authority_freeze, phase="POST_CAPTURE")
        _verify_file_freeze(parent_table_freeze, phase="POST_CAPTURE")
    source_paths = successor_support_evidence_paths(source_root)
    source_storage_bytes = sum(path.stat().st_size for path in source_paths.values())
    source_table_bytes = sum(
        int(dict(ref)["byte_length"])
        for ref in dict(source_manifest["table_files"]).values()
    )
    post_capture_resource = _resource_preflight_receipt(
        phase="POST_CAPTURE",
        source_root=source_root,
        staging_root=staging,
        canonical_root=canonical,
        parent_storage_bytes=parent_storage_bytes,
        planned_request_count=int(source_plan["planned_request_count"]),
        source_storage_bytes=source_storage_bytes,
        source_table_bytes=source_table_bytes,
        predecessor_receipt_sha256=pre_provider_resource["receipt_sha256"],
    )
    if taint_analysis_dry_run:
        assert audit_root is not None
        authority_bindings = {
            "predecessor_pointer": {
                "path": str(canonical / FUNDAMENTAL_POINTER_FILENAME),
                "sha256": _sha256(parent_pointer_bytes),
            },
            "predecessor_manifest": {
                "path": str(parent_manifest_path),
                "sha256": _sha256(parent_manifest_bytes),
            },
            "market_pointer": {
                "path": str(market_pointer_path),
                "sha256": _sha256(market_pointer_bytes),
            },
            "pit_pointer": {
                "path": str(pit_pointer_path),
                "sha256": _sha256(pit_pointer_bytes),
            },
            "pit_membership": {
                "path": str(membership_path),
                "sha256": _sha256(membership_bytes),
            },
            "target_scope": {
                "path": str(scope_path),
                "sha256": _sha256(scope_bytes),
            },
            "history_audit": {
                "path": str(history_path),
                "sha256": _sha256(history_bytes),
            },
        }
        analysis = analyze_deferred_fundamental_taints(
            fileset_root=source_root,
            parent_period_path=parent_tables["fundamental_period"],
            parent_daily_path=parent_tables["fundamental_daily"],
            membership_path=membership_path,
            parent_cutoff=parent_cutoff,
            target_cutoff=target,
            support_start=support_start,
            authority_bindings=authority_bindings,
        )
        analysis_root = audit_root / "analysis"
        if not analysis_root.exists():
            analysis_root.mkdir(mode=0o700)
        os.chmod(analysis_root, 0o700)
        _atomic_private_json(
            analysis_root / "non_reachability_report.json",
            analysis["report"],
        )
        _atomic_private_json(
            analysis_root / "source_analysis_closure.json",
            analysis["source_analysis_closure"],
        )
        final_authority = _verify_authority_freeze(
            authority_freeze,
            phase="FINAL_RECEIPT",
        )
        final_authority.update(
            _verify_file_freeze(
                parent_table_freeze,
                phase="FINAL_RECEIPT",
            )
        )
        report = dict(analysis["report"])
        closure = dict(analysis["source_analysis_closure"])
        status = str(report["taint_analysis_status"])
        receipt = _sealed(
            {
                "schema_version": TAINT_DRY_RUN_EXECUTION_SCHEMA,
                "taint_analysis_status": status,
                "maintenance_status": (
                    "diagnostic_pass" if status == "PASS" else "blocked"
                ),
                "run_id": generation_id,
                "parent_generation_id": parent_pointer.get("generation_id"),
                "parent_cutoff": parent_cutoff,
                "target_cutoff": target,
                "open_sessions": sessions,
                "support_start": support_start,
                "live_provider_capture": True,
                "canonical_write": False,
                "canonical_pointer_unchanged": True,
                "staging_written": False,
                "promotion_authorized": False,
                "factor_or_mainline_changed": False,
                "authoritative_source_ready": False,
                "staging_eligible": False,
                "promotion_eligible": False,
                "canonical_write_authorized": False,
                "usable_for_investment_research": False,
                "source_fileset_manifest_sha256": source_manifest[
                    "manifest_sha256"
                ],
                "taint_report_sha256": report["report_sha256"],
                "source_analysis_closure_sha256": closure[
                    "closure_sha256"
                ],
                "deferred_observation_count": report[
                    "deferred_observation_count"
                ],
                "tainted_non_reachable_count": report[
                    "tainted_non_reachable_count"
                ],
                "blocking_unknown_count": report[
                    "blocking_unknown_count"
                ],
                "provider_accounting": dict(
                    source_manifest["provider_accounting"]
                ),
                "planned_request_count": int(
                    source_plan["planned_request_count"]
                ),
                "dependency_contract_sha256": (
                    SUCCESSOR_FINANCIAL_DEPENDENCY_CONTRACT_SHA256
                ),
                "implementation_sha256": implementation_sha256,
                "final_authority_sha256": final_authority,
                "pre_provider_resource_preflight_sha256": (
                    pre_provider_resource["receipt_sha256"]
                ),
                "post_capture_resource_preflight_sha256": (
                    post_capture_resource["receipt_sha256"]
                ),
                "audit_run_root": str(audit_root),
                "write_set": [
                    str(audit_root / "frozen_state.json"),
                    str(source_root),
                    str(analysis_root / "non_reachability_report.json"),
                    str(analysis_root / "source_analysis_closure.json"),
                    str(audit_root / "execution_receipt.json"),
                ],
                "write_boundaries": {
                    "analysis": False,
                    "candidates": False,
                    "portfolio": False,
                    "factor_or_mainline": False,
                    "dashboard": False,
                    "paper": False,
                    "broker": False,
                    "orders": False,
                    "trades": False,
                },
            },
            field="execution_receipt_sha256",
        )
        _atomic_private_json(
            audit_root / "execution_receipt.json",
            receipt,
        )
        return receipt
    derivation_resource_budget = _derivation_resource_budget(
        post_capture_receipt=post_capture_resource,
        source_manifest=source_manifest,
        keyset=keyset,
    )
    captured_financial_support, financial_absence_proofs = _financial_support_closure(
        source_plan=source_plan,
        source_manifest=source_manifest,
        target_cutoff=target,
    )
    raw_tables = open_support_tables(source_root)
    source_evidence: dict[str, bytes | Path] = {
        f"source/{name}": path for name, path in source_paths.items()
    }
    authority_evidence = {
        "authority/predecessor_pointer.json": parent_pointer_bytes,
        "authority/market_pointer.json": market_pointer_bytes,
        "authority/pit_pointer.json": pit_pointer_bytes,
        "authority/predecessor_manifest.json": parent_manifest_bytes,
        "authority/market_manifest.json": market_manifest_bytes,
        "authority/pit_manifest.json": pit_manifest_bytes,
        "authority/pit_membership.parquet": membership_bytes,
        "authority/expected_scope.json": scope_bytes,
        "authority/canonical_subject_scope.json": subject_scope_bytes,
        "authority/history_audit.json": history_bytes,
        "authority/pre_provider_resource_preflight.json": _canonical_json_bytes(
            pre_provider_resource
        ),
        "authority/post_capture_resource_preflight.json": _canonical_json_bytes(
            post_capture_resource
        ),
        "authority/derivation_resource_budget.json": _canonical_json_bytes(
            derivation_resource_budget
        ),
    }
    historical_taint_registry: dict[str, Any] | None = None
    historical_taint_evidence_files: dict[str, bytes | Path] = {}
    if append_first:
        predecessor_reference = dict(
            build_successor_chain(
                parent_closure,
                parent_cutoff=parent_cutoff,
                target_cutoff=target,
                generation_id=generation_id,
            )["immediate_predecessor"]
        )
        (
            historical_taint_registry,
            historical_paths,
        ) = build_historical_taint_registry(
            failure_evidence=historical_taint_evidence,
            predecessor=predecessor_reference,
            parent_cutoff=parent_cutoff,
            target_cutoff=target,
            delta_fileset_root=source_root,
        )
        historical_taint_evidence_files = {
            name: path for name, path in historical_paths.items()
        }
        historical_taint_evidence_files[
            "historical_taint/registry.json"
        ] = _canonical_json_bytes(historical_taint_registry)
    evidence = {
        **source_evidence,
        **authority_evidence,
        **historical_taint_evidence_files,
    }
    evidence_sha: dict[str, str] = {}
    for name, value in evidence.items():
        if isinstance(value, bytes):
            evidence_sha[name] = _sha256(value)
        else:
            evidence_sha[name], _size = _stable_regular_file_sha256(
                value,
                label=f"provider evidence {name}",
            )
    permanent_refs = _relative_support_refs(evidence_sha)
    plan_metadata = seal_support_plan(
        raw_tables,
        parent_cutoff=parent_cutoff,
        target_cutoff=target,
        permanent_support_refs=permanent_refs,
        support_prefix_mode=(
            SUCCESSOR_APPEND_FIRST_MODE if append_first else "validation_only"
        ),
        absence_proofs=financial_absence_proofs,
        historical_taint_registry_sha256=(
            str(historical_taint_registry["registry_sha256"])
            if historical_taint_registry is not None
            else ""
        ),
        append_first_financial_dependencies=captured_financial_support,
        extra={
            "support_provider_contract": str(source_manifest.get("schema_version") or ""),
            "source_fileset_manifest_sha256": str(
                source_manifest.get("manifest_sha256") or ""
            ),
            "source_plan_sha256": str(source_plan.get("plan_sha256") or ""),
            "support_start": support_start,
            "implementation_sha256": implementation_sha256,
            "pre_provider_resource_preflight_sha256": pre_provider_resource[
                "receipt_sha256"
            ],
            "post_capture_resource_preflight_sha256": post_capture_resource[
                "receipt_sha256"
            ],
            "derivation_resource_budget": derivation_resource_budget,
            "append_first": append_first,
        },
    )
    bundle = assemble_safe_successor(
        parent_tables=parent_tables,
        parent_closure=parent_closure,
        support_raw_tables=raw_tables,
        plan_metadata=plan_metadata,
        keyset_closure=keyset,
        parent_cutoff=parent_cutoff,
        target_cutoff=target,
        run_id=generation_id,
        staging_parent=staging.parent,
    )
    request_receipts_sha256 = canonical_json_sha256(
        {"request_receipts": list(source_manifest.get("request_receipts", []) or [])}
    )
    provider_manifest = seal_successor_provider_manifest(
        bundle,
        provider="tushare_official_https",
        request_receipts_sha256=request_receipts_sha256,
        evidence_files=evidence_sha,
        extra={
            "source_fileset_manifest_sha256": str(
                source_manifest.get("manifest_sha256") or ""
            ),
            "source_binding_sha256": str(
                dict(source_manifest.get("binding", {}) or {}).get(
                    "binding_sha256", ""
                )
            ),
            "source_provider_accounting": dict(
                source_manifest.get("provider_accounting", {}) or {}
            ),
            "implementation_sha256": implementation_sha256,
            "pre_provider_resource_preflight": pre_provider_resource,
            "post_capture_resource_preflight": post_capture_resource,
            "captured_pointer_sha256": {
                "predecessor": _sha256(parent_pointer_bytes),
                "market": _sha256(market_pointer_bytes),
                "pit": _sha256(pit_pointer_bytes),
            },
            "append_first": append_first,
            "historical_taint_registry_sha256": (
                str(historical_taint_registry["registry_sha256"])
                if historical_taint_registry is not None
                else ""
            ),
            "historical_taint_registry_file_sha256": (
                evidence_sha.get("historical_taint/registry.json", "")
            ),
        },
    )
    target_bindings = {
        "market_pointer": {
            "path": str(market_pointer_path),
            "sha256": _sha256(market_pointer_bytes),
            "as_of": target,
            "immutable_refs": [
                _immutable_ref(market_manifest_path, market_manifest_bytes),
                _immutable_ref(scope_path, scope_bytes),
                _immutable_ref(history_path, history_bytes),
            ],
        },
        "pit_pointer": {
            "path": str(pit_pointer_path),
            "sha256": _sha256(pit_pointer_bytes),
            "as_of": target,
            "immutable_refs": [
                _immutable_ref(pit_manifest_path, pit_manifest_bytes),
                _immutable_ref(membership_path, membership_bytes),
            ],
        },
        "pit_membership": {
            "path": str(membership_path),
            "sha256": _sha256(membership_bytes),
            "as_of": target,
        },
        "expected_scope": {
            "path": str(scope_path),
            "sha256": _sha256(scope_bytes),
            "as_of": target,
        },
    }
    capture = stage_successor_generation(
        bundle,
        staging_root=staging,
        generation_id=generation_id,
        provider_manifest=provider_manifest,
        target_bindings=target_bindings,
        provider_evidence_files=evidence,
        metadata_extra={
            "maintenance_schema_version": SAFE_SUCCESSOR_MAINTENANCE_SCHEMA,
            "target_classification_counts": target_counts,
            "target_market_scope_classification_counts": coverage_target_counts,
            "canonical_subject_scope_sha256": subject_scope_closure[
                "closure_sha256"
            ],
            "support_fileset_root": str(source_root),
            "pre_provider_resource_preflight_sha256": pre_provider_resource[
                "receipt_sha256"
            ],
            "post_capture_resource_preflight_sha256": post_capture_resource[
                "receipt_sha256"
            ],
            "append_first": append_first,
        },
    )
    accounting = dict(source_manifest.get("provider_accounting", {}) or {})
    return {
        "schema_version": SAFE_SUCCESSOR_MAINTENANCE_SCHEMA,
        "status": "staged",
        "maintenance_status": "staged_pending_promotion",
        "canonical_pointer_unchanged": True,
        "generation_id": capture.generation_id,
        "parent_generation_id": parent_pointer.get("generation_id"),
        "parent_cutoff": parent_cutoff,
        "target_cutoff": target,
        "original_seam": dict(bundle.successor_chain.get("original_seam", {}) or {}).get(
            "cutoff", parent_cutoff
        ),
        "open_sessions": sessions,
        "support_start": support_start,
        "append_first": append_first,
        "historical_taint_registry_sha256": (
            str(historical_taint_registry["registry_sha256"])
            if historical_taint_registry is not None
            else ""
        ),
        "historical_taint_count": (
            int(historical_taint_registry["historical_conflict_count"])
            if historical_taint_registry is not None
            else 0
        ),
        "income_support_dependency_count": len(
            [
                value
                for value in list(
                    source_plan.get("financial_support_dependencies", []) or []
                )
                if value.get("table") == "income"
            ]
        ),
        "financial_support_dependency_count": len(
            list(source_plan.get("financial_support_dependencies", []) or [])
        ),
        "financial_support_row_dependency_count": len(captured_financial_support),
        "financial_support_absence_proof_count": len(financial_absence_proofs),
        "provider_accounting": accounting,
        "planned_request_count": int(source_plan.get("planned_request_count", 0)),
        "raw_row_counts": {
            name: int(dict(ref)["row_count"])
            for name, ref in dict(source_manifest["table_files"]).items()
        },
        "suffix_rows": {
            "fundamental_period": len(bundle.period_suffix),
            "fundamental_daily": len(bundle.daily_suffix),
            "fundamental_quarantine": 0,
        },
        "classification_counts": target_counts,
        "market_scope_classification_counts": coverage_target_counts,
        "canonical_subject_count": len(symbols),
        "target_market_scope_count": len(target_scope_symbols),
        "canonical_subject_scope_sha256": subject_scope_closure[
            "closure_sha256"
        ],
        "staging_root": str(capture.staging_root),
        "staging_pointer_path": str(capture.pointer_path),
        "staging_pointer_sha256": capture.pointer_sha256,
        "staging_manifest_path": str(capture.manifest_path),
        "staging_manifest_sha256": capture.manifest_sha256,
        "staging_table_sha256": dict(capture.table_sha256),
        "predecessor_pointer_sha256": str(parent_closure["pointer_sha256"]),
        "market_pointer_sha256": _sha256(market_pointer_bytes),
        "pit_pointer_sha256": _sha256(pit_pointer_bytes),
        "source_fileset_root": str(source_root),
        "source_fileset_manifest_sha256": str(
            source_manifest.get("manifest_sha256") or ""
        ),
        "pre_provider_resource_preflight": pre_provider_resource,
        "post_capture_resource_preflight": post_capture_resource,
        "gate2_contract": str(bundle.readiness.get("schema_version") or ""),
        "gate2_passed": bundle.readiness.get("gate2_passed") is True,
        "machine_states": {
            "mixed": True,
            "legacy_direct_reader_provenance": "limited",
            "binding_aware_research_ready": True,
            "homogeneous_history_ready": False,
        },
        "write_boundaries": {
            "analysis": False,
            "candidates": False,
            "portfolio": False,
            "factor_or_mainline": False,
            "dashboard": False,
            "paper": False,
            "broker": False,
            "orders": False,
            "trades": False,
        },
    }


def run_cn_fundamental_taint_dry_run(
    *,
    as_of: str,
    run_id: str,
    audit_run_root: str | Path,
    canonical_root: str | Path,
    expected_pointer_sha256: str,
    canonical_market_pointer_path: str | Path,
    canonical_pit_pointer_path: str | Path,
    canonical_membership_path: str | Path,
    canonical_scope_path: str | Path,
    history_audit_path: str | Path,
    expected_history_audit_sha256: str,
    allow_live: bool,
    universes: Sequence[str] | str,
    max_attempts: int = 3,
    retry_backoff_seconds: Sequence[float] = (0.5, 1.0),
    requests_per_second: float = 8.0,
    client: Any | None = None,
) -> dict[str, Any]:
    """Run the exclusive live-capture, zero-staging taint diagnostic."""

    audit = Path(audit_run_root).expanduser()
    if not audit.is_absolute():
        _fail("TAINT_AUDIT_ROOT_MUST_BE_ABSOLUTE")
    try:
        return run_cn_fundamental_safe_successor(
            as_of=as_of,
            run_id=run_id,
            staging_root=audit / "_staging_forbidden",
            canonical_root=canonical_root,
            expected_pointer_sha256=expected_pointer_sha256,
            canonical_market_pointer_path=canonical_market_pointer_path,
            canonical_pit_pointer_path=canonical_pit_pointer_path,
            canonical_membership_path=canonical_membership_path,
            canonical_scope_path=canonical_scope_path,
            history_audit_path=history_audit_path,
            expected_history_audit_sha256=expected_history_audit_sha256,
            support_fileset_root=audit / "capture",
            allow_live=allow_live,
            universes=universes,
            max_attempts=max_attempts,
            retry_backoff_seconds=retry_backoff_seconds,
            requests_per_second=requests_per_second,
            client=client,
            taint_analysis_dry_run=True,
            audit_run_root=audit,
        )
    except Exception as exc:
        if not audit.exists() or not audit.is_dir():
            raise
        code = str(getattr(exc, "code", "TAINT_DRY_RUN_FAILED"))
        pointer_path = Path(canonical_root) / FUNDAMENTAL_POINTER_FILENAME
        pointer_unchanged = False
        try:
            pointer_unchanged = _sha256(
                _stable_regular_bytes(
                    pointer_path,
                    label="blocked receipt Fundamental pointer",
                )
            ) == str(expected_pointer_sha256).strip().lower()
        except Exception:
            pointer_unchanged = False
        receipt = _sealed(
            {
                "schema_version": TAINT_DRY_RUN_EXECUTION_SCHEMA,
                "taint_analysis_status": "BLOCKED",
                "maintenance_status": "blocked",
                "blocker_codes": [code],
                "run_id": str(run_id),
                "target_cutoff": str(as_of),
                "live_provider_capture": True,
                "canonical_pointer_unchanged": pointer_unchanged,
                "canonical_write": False,
                "staging_written": False,
                "promotion_authorized": False,
                "factor_or_mainline_changed": False,
                "authoritative_source_ready": False,
                "staging_eligible": False,
                "promotion_eligible": False,
                "canonical_write_authorized": False,
                "usable_for_investment_research": False,
                "audit_run_root": str(audit),
                "write_set": [str(audit)],
            },
            field="execution_receipt_sha256",
        )
        _atomic_private_json(audit / "execution_receipt.json", receipt)
        return receipt


__all__ = [
    "FundamentalSuccessorMaintenanceError",
    "SAFE_SUCCESSOR_MAINTENANCE_SCHEMA",
    "TAINT_DRY_RUN_EXECUTION_SCHEMA",
    "run_cn_fundamental_safe_successor",
    "run_cn_fundamental_taint_dry_run",
]
