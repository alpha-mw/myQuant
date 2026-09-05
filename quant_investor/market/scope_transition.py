"""Explicit, receipt-bound full-A scope transitions in the daily maintenance path.

Default readers fail closed while an operation is incomplete. Only the locked
operation's process-local context can read its partially committed inputs.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping

SCHEMA = "cn-scope-transition-request.v1"
_OWNER: ContextVar[str] = ContextVar("cn_scope_transition_owner", default="")


def _bytes(path: str | Path) -> bytes:
    from .pit_universe import _stable_read_bytes

    p = Path(path)
    if not p.is_absolute() or p.is_symlink():
        raise RuntimeError("SCOPE_TRANSITION_PATH_INVALID")
    return _stable_read_bytes(p, blocker="SCOPE_TRANSITION_FILE_CHANGED")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def reference(path: str | Path) -> dict[str, str]:
    return {"path": str(path), "sha256": digest(_bytes(path))}


def read_ref(value: Mapping[str, Any]) -> bytes:
    if set(value) != {"path", "sha256"}:
        raise RuntimeError("SCOPE_TRANSITION_REFERENCE_INVALID")
    raw = _bytes(value["path"])
    if digest(raw) != value["sha256"]:
        raise RuntimeError("SCOPE_TRANSITION_REFERENCE_SHA_MISMATCH")
    return raw


def encoded(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _write_new(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists():
        if _bytes(path) != raw:
            raise RuntimeError("SCOPE_TRANSITION_IMMUTABLE_CONFLICT")
        return
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _replace(path: Path, raw: bytes) -> None:
    from .pit_universe import _atomic_write_bytes

    _atomic_write_bytes(path, raw)


def marker_path(workspace: Path) -> Path:
    return workspace / "data/private/cn_daily_maintenance/scope_transition_active.json"


def assert_scope_readable(workspace: str | Path) -> None:
    root = Path(workspace).resolve()
    marker = marker_path(root)
    if marker.exists() or marker.is_symlink():
        payload = json.loads(_bytes(marker))
        if _OWNER.get() != payload.get("request_sha256"):
            raise RuntimeError("SCOPE_TRANSITION_IN_PROGRESS")


@contextmanager
def _owned(request_sha: str):
    token = _OWNER.set(request_sha)
    try:
        yield
    finally:
        _OWNER.reset(token)


def load_request(path: str | Path, expected_sha: str) -> dict[str, Any]:
    raw = _bytes(path)
    mode = os.stat(path).st_mode
    if stat.S_IMODE(mode) != 0o600 or os.stat(path).st_uid != os.getuid():
        raise RuntimeError("SCOPE_TRANSITION_REQUEST_NOT_OWNER_ONLY")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha) or digest(raw) != expected_sha:
        raise RuntimeError("SCOPE_TRANSITION_REQUEST_SHA_MISMATCH")
    q = json.loads(raw)
    required = {
        "schema_version",
        "operation_id",
        "workspace_root",
        "effective_date",
        "canonical_scope_path",
        "old_scope_ref",
        "new_scope_ref",
        "pit_capture_ref",
        "close_receipt_ref",
        "pit_pointer_ref",
        "market_pointer_ref",
        "fundamental_pointer_ref",
        "checkpoint_ref",
        "veto_ref",
        "failed_attempt_ref",
        "added",
        "removed",
        "authority",
    }
    if set(q) != required or q["schema_version"] != SCHEMA:
        raise RuntimeError("SCOPE_TRANSITION_REQUEST_INVALID")
    if q["authority"] != "OWNER_AUTHORIZED_DATA_SCOPE_ONLY":
        raise RuntimeError("SCOPE_TRANSITION_AUTHORITY_INVALID")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", q["operation_id"]):
        raise RuntimeError("SCOPE_TRANSITION_ID_INVALID")
    root = Path(q["workspace_root"])
    if not root.is_absolute() or root.resolve() != root:
        raise RuntimeError("SCOPE_TRANSITION_WORKSPACE_INVALID")
    fixed = {
        "canonical_scope_path": root / "data/cn_universe/cn_index_components.json",
        "pit_pointer_ref": root / "data/parquet/cn/reference/stock_basic_membership_latest.json",
        "market_pointer_ref": root / "data/parquet/cn/_latest.json",
        "fundamental_pointer_ref": root / "data/parquet/cn/_fundamental_latest.json",
        "veto_ref": root / "data/private/cn_daily_maintenance/WRITE_VETO.json",
    }
    for key, expected in fixed.items():
        value = q[key] if key == "canonical_scope_path" else q[key]["path"]
        if value != str(expected):
            raise RuntimeError("SCOPE_TRANSITION_CANONICAL_PATH_MISMATCH")
    for key, value in q.items():
        if key.endswith("_ref"):
            p = Path(value["path"])
            if not p.is_relative_to(root) or p.resolve() != p:
                raise RuntimeError("SCOPE_TRANSITION_REFERENCE_OUTSIDE_WORKSPACE")
    if not Path(q["checkpoint_ref"]["path"]).is_relative_to(root / "data/staging"):
        raise RuntimeError("SCOPE_TRANSITION_CHECKPOINT_PATH_INVALID")
    old = json.loads(read_ref(q["old_scope_ref"]))["full_a"]
    new = json.loads(read_ref(q["new_scope_ref"]))["full_a"]
    for values in (old, new):
        if (
            not values
            or len(values) != len(set(values))
            or any(not re.fullmatch(r"[0-9]{6}\.(SH|SZ|BJ)", x) for x in values)
        ):
            raise RuntimeError("SCOPE_TRANSITION_SYMBOLS_INVALID")
    if q["added"] != sorted(set(new) - set(old)) or q["removed"] != sorted(set(old) - set(new)):
        raise RuntimeError("SCOPE_TRANSITION_DELTA_MISMATCH")
    capture = json.loads(read_ref(q["pit_capture_ref"]))
    close = json.loads(read_ref(q["close_receipt_ref"]))
    if (
        capture["effective_date"] != q["effective_date"]
        or close["target_trade_date"] != q["effective_date"]
        or close["status"] != "TARGET_AUTHORIZED"
        or close["endpoint_url"] != "https://api.tushare.pro/"
    ):
        raise RuntimeError("SCOPE_TRANSITION_EFFECTIVE_DATE_MISMATCH")
    raw_calendar = _bytes(close["raw_response_path"])
    if digest(raw_calendar) != close["raw_response_sha256"]:
        raise RuntimeError("SCOPE_TRANSITION_CALENDAR_SHA_MISMATCH")
    rows: dict[str, dict[str, Any]] = {}
    for part in capture["partitions"]:
        payload = json.loads(read_ref({"path": part["path"], "sha256": part["sha256"]}))
        if len(payload["items"]) != part["row_count"]:
            raise RuntimeError("SCOPE_TRANSITION_CAPTURE_COUNT_MISMATCH")
        for item in payload["items"]:
            rows[item["ts_code"]] = item
    if set(new) != {s for s, item in rows.items() if item["list_status"] == "L"}:
        raise RuntimeError("SCOPE_TRANSITION_LISTED_SET_MISMATCH")
    for s in q["added"]:
        d = rows[s].get("list_date") or ""
        if not re.fullmatch(r"[0-9]{8}", d) or d > q["effective_date"]:
            raise RuntimeError("SCOPE_TRANSITION_ADDITION_NOT_EFFECTIVE")
    for s in q["removed"]:
        item = rows.get(s, {})
        d = item.get("delist_date") or ""
        if (
            item.get("list_status") != "D"
            or not re.fullmatch(r"[0-9]{8}", d)
            or d > q["effective_date"]
        ):
            raise RuntimeError("SCOPE_TRANSITION_REMOVAL_NOT_EFFECTIVE")
    failed = json.loads(read_ref(q["failed_attempt_ref"]))
    veto_path = Path(q["veto_ref"]["path"])
    veto_source = (
        q["veto_ref"]
        if veto_path.exists()
        else {
            "path": str(veto_path.parent / "veto_archive" / (q["veto_ref"]["sha256"] + ".json")),
            "sha256": q["veto_ref"]["sha256"],
        }
    )
    veto = json.loads(read_ref(veto_source))
    expected_blockers = ["PIT_CONTRACT_BLOCKED", "UPSTREAM_STAGE_NOT_READY"]
    if (
        veto.get("schema_version") != "cn-daily-maintenance-write-veto.v1"
        or veto.get("target_date") != q["effective_date"]
        or veto.get("attempt_slot") != "2020"
        or veto.get("attempt_root") != str(Path(q["failed_attempt_ref"]["path"]).parent)
        or veto.get("blockers") != expected_blockers
        or failed.get("target_date") != q["effective_date"]
        or failed.get("attempt_slot") != "2020"
        or failed.get("mode") != "execute"
        or failed.get("blockers") != expected_blockers
        or failed.get("close_session_receipt_ref") != q["close_receipt_ref"]
        or any(row.get("write_performed") is not False for row in failed.get("stage_results", []))
        or failed.get("canonical_write_count") != 0
        or failed.get("canonical_unchanged") is not True
        or failed.get("write_veto_ref") != q["veto_ref"]
        or failed.get("stage_results", [{}])[0].get("evidence", {}).get("error_code")
        != "pit_frozen_scope_predecessor_binding_changed"
    ):
        raise RuntimeError("SCOPE_TRANSITION_SOURCE_FAILURE_INVALID")
    return q


def validate_pit_transition(
    q: Mapping[str, Any],
    *,
    parent: Mapping[str, Any],
    scope_path: str,
    scope_sha: str,
    capture_sha: str,
) -> None:
    parent_scope = dict(parent.get("manifest", {}).get("source_bindings", {})).get("full_a_scope")
    if not isinstance(parent_scope, Mapping):
        raise RuntimeError("SCOPE_TRANSITION_PREDECESSOR_MISMATCH")
    if (
        scope_path not in {q["canonical_scope_path"], q["new_scope_ref"]["path"]}
        or scope_sha != q["new_scope_ref"]["sha256"]
        or capture_sha != q["pit_capture_ref"]["sha256"]
        or parent["discovery_pointer_sha256"] != q["pit_pointer_ref"]["sha256"]
        or parent_scope["sha256"] != q["old_scope_ref"]["sha256"]
        or parent_scope["path"] != q["canonical_scope_path"]
    ):
        raise RuntimeError("SCOPE_TRANSITION_PREDECESSOR_MISMATCH")
    old_records = {x.symbol: x for x in parent["records"]}
    capture = json.loads(read_ref(q["pit_capture_ref"]))
    listed = json.loads(_bytes(capture["partitions"][0]["path"]))["items"]
    new_records = {x["ts_code"]: x for x in listed}
    for symbol in q["added"]:
        previous = old_records.get(symbol)
        if previous and (
            previous.membership_quality != "outside_frozen_scope_pending"
            or previous.list_date != new_records[symbol]["list_date"]
            or previous.source_list_status != "L"
        ):
            raise RuntimeError("SCOPE_TRANSITION_PENDING_IDENTITY_DRIFT")


def require_transition_publisher(request_path: str | Path, request_sha: str) -> None:
    q = load_request(request_path, request_sha)
    marker = marker_path(Path(q["workspace_root"]))
    if (
        _OWNER.get() != request_sha
        or not marker.exists()
        or json.loads(_bytes(marker)).get("request_sha256") != request_sha
        or _bytes(q["canonical_scope_path"]) != read_ref(q["new_scope_ref"])
    ):
        raise RuntimeError("SCOPE_TRANSITION_PUBLICATION_REQUIRES_LOCKED_OPERATION")


def _verify_closure(q: Mapping[str, Any]) -> dict[str, Any]:
    from .daily_components import _market_reference
    from .pit_universe import PITUniverseStore
    from .market_data_store import MarketDataStore

    root = Path(q["workspace_root"])
    market = _market_reference(root / "data")
    pit = PITUniverseStore(root_dir=root / "data/parquet/cn/reference").load_generation_binding()
    scope = json.loads(_bytes(q["canonical_scope_path"]))["full_a"]
    cov = market["pointer"]["coverage"]
    if (
        digest(_bytes(q["canonical_scope_path"])) != q["new_scope_ref"]["sha256"]
        or cov["expected_scope_count"] != len(scope)
        or cov["expected_scope_sha256"] != digest("\n".join(sorted(scope)).encode())
        or cov["pit_membership_sha256"] != pit["canonical_sha256"]
        or cov["pit_generation_manifest_sha256"] != pit["generation_manifest_sha256"]
        or market["pointer"]["latest_complete_trade_date"] != q["effective_date"]
        or pit["manifest"]["source_bindings"]["full_a_scope"]["sha256"]
        != q["new_scope_ref"]["sha256"]
    ):
        raise RuntimeError("SCOPE_TRANSITION_FINAL_BINDING_MISMATCH")
    validation = MarketDataStore(market="CN", data_root=root / "data").validate_latest()
    if validation.get("status") != "passed":
        raise RuntimeError("SCOPE_TRANSITION_STORAGE_VALIDATION_FAILED")
    read_ref(q["fundamental_pointer_ref"])
    read_ref(q["checkpoint_ref"])
    return {
        "scope_ref": reference(q["canonical_scope_path"]),
        "scope_count": len(scope),
        "effective_date": q["effective_date"],
        "market_pointer_ref": reference(market["pointer_path"]),
        "market_manifest_ref": reference(market["snapshot_manifest_path"]),
        "pit_pointer_ref": reference(pit["discovery_pointer_path"]),
        "pit_membership_ref": reference(pit["canonical_path"]),
        "pit_manifest_ref": reference(pit["generation_manifest_path"]),
        "storage_validation": validation,
    }


def _recover_exact_veto(q: Mapping[str, Any], op: Path, request_sha: str) -> dict[str, Any]:
    """Finish the existing exact archive path only after a verified terminal."""
    from .daily_maintenance import _clear_cn_daily_write_veto_locked

    terminal_ref = reference(op / "terminal.json")
    run = Path(q["workspace_root"]) / "data/private/cn_daily_maintenance"
    archive = run / "veto_archive" / (q["veto_ref"]["sha256"] + ".json")
    record = op / "veto-recovery.json"
    intent = {
        "request_sha256": request_sha,
        "terminal_ref": terminal_ref,
        "veto_ref": q["veto_ref"],
    }
    intent_path = op / "veto-archive-intent.json"
    if record.exists():
        recovery = json.loads(_bytes(record))
        if (
            set(recovery)
            != {
                "schema_version",
                "status",
                "request_sha256",
                "terminal_ref",
                "archived_veto_ref",
                "clear_receipt_ref",
                "recovery_observed_at",
                "recovery_mode",
            }
            or recovery["schema_version"] != "cn-scope-transition-veto-recovery.v1"
            or recovery["status"] != "RECOVERED"
            or recovery["request_sha256"] != request_sha
            or recovery["terminal_ref"] != terminal_ref
            or recovery["archived_veto_ref"]
            != {"path": str(archive), "sha256": q["veto_ref"]["sha256"]}
        ):
            raise RuntimeError("SCOPE_TRANSITION_VETO_RECOVERY_DRIFT")
        if not intent_path.exists() or json.loads(_bytes(intent_path)) != intent:
            raise RuntimeError("SCOPE_TRANSITION_VETO_INTENT_DRIFT")
        read_ref(recovery["archived_veto_ref"])
        if recovery["clear_receipt_ref"] is None:
            if recovery["recovery_mode"] != "ARCHIVE_ONLY_RECOVERY":
                raise RuntimeError("SCOPE_TRANSITION_VETO_RECOVERY_MODE_INVALID")
        else:
            clear = json.loads(read_ref(recovery["clear_receipt_ref"]))
            if (
                recovery["recovery_mode"] != "CLEAR_RECEIPT_VERIFIED"
                or set(clear)
                != {
                    "schema_version",
                    "status",
                    "cleared",
                    "cleared_at",
                    "reason",
                    "lane",
                    "archived_veto_ref",
                }
                or clear["schema_version"] != "cn-daily-maintenance-veto-clear.v1"
                or clear["status"] != "CLEARED"
                or clear["cleared"] is not True
                or clear["reason"] != f"SCOPE_TRANSITION_VERIFIED:{request_sha}"
                or clear["lane"] != "global"
                or clear["archived_veto_ref"] != recovery["archived_veto_ref"]
            ):
                raise RuntimeError("SCOPE_TRANSITION_VETO_CLEAR_RECEIPT_INVALID")
        if Path(q["veto_ref"]["path"]).exists():
            raise RuntimeError("SCOPE_TRANSITION_VETO_REAPPEARED")
        return recovery
    if Path(q["veto_ref"]["path"]).exists():
        read_ref(q["veto_ref"])
        _write_new(intent_path, encoded(intent))
        cleared = _clear_cn_daily_write_veto_locked(
            run_root=run,
            expected_veto_sha256=q["veto_ref"]["sha256"],
            reason=f"SCOPE_TRANSITION_VERIFIED:{request_sha}",
        )
        if cleared["status"] != "CLEARED":
            raise RuntimeError("SCOPE_TRANSITION_VETO_NOT_CLEARED")
        read_ref(cleared["clear_receipt_ref"])
        clear_ref = cleared["clear_receipt_ref"]
    else:
        # A vanished veto alone is never evidence. Both pre-clear intent and
        # exact archived bytes are mandatory for post-archive crash recovery.
        if not intent_path.exists() or json.loads(_bytes(intent_path)) != intent:
            raise RuntimeError("SCOPE_TRANSITION_VETO_MISSING_WITHOUT_RECOVERY")
        clear_ref = None
    archived = {"path": str(archive), "sha256": q["veto_ref"]["sha256"]}
    read_ref(archived)
    recovery = {
        "schema_version": "cn-scope-transition-veto-recovery.v1",
        "status": "RECOVERED",
        "request_sha256": request_sha,
        "terminal_ref": terminal_ref,
        "archived_veto_ref": archived,
        "clear_receipt_ref": clear_ref,
        "recovery_mode": "CLEAR_RECEIPT_VERIFIED" if clear_ref else "ARCHIVE_ONLY_RECOVERY",
        "recovery_observed_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_new(record, encoded(recovery))
    return recovery


def _ready_receipt(op: Path) -> dict[str, Any]:
    terminal = json.loads(_bytes(op / "terminal.json"))
    ready = {
        **terminal,
        "status": "VERIFIED",
        "claude_input_ready": True,
        "terminal_ref": reference(op / "terminal.json"),
        "veto_recovery_ref": reference(op / "veto-recovery.json"),
    }
    _write_new(op / "readiness.json", encoded(ready))
    return {**ready, "receipt_ref": reference(op / "readiness.json")}


def _retire_stale_coverage_declaration(q, op: Path, expected_sha: str) -> dict[str, Any]:
    """Archive an explicitly selected obsolete per-run exemption, never rebind it."""
    from .cn_nontrading_evidence import canonical_json_sha256

    if _OWNER.get() != op.name:
        raise RuntimeError("SCOPE_COVERAGE_RETIREMENT_REQUIRES_OPERATION")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise RuntimeError("SCOPE_COVERAGE_DECLARATION_SHA_INVALID")
    source = Path(q["workspace_root"]) / "data/cn_universe/daily_basic_coverage_boundaries.json"
    archive = op / "retired-inputs" / f"daily-basic-coverage-{expected_sha}.json"
    receipt = op / f"coverage-declaration-retirement-{expected_sha}.json"
    replayed = receipt.exists()
    if replayed:
        result = json.loads(_bytes(receipt))
        if (
            result.get("request_sha256") != op.name
            or result.get("archived_ref") != {"path": str(archive), "sha256": expected_sha}
            or result.get("readiness_ref") != reference(op / "readiness.json")
            or source.exists()
        ):
            raise RuntimeError("SCOPE_COVERAGE_RETIREMENT_REPLAY_DRIFT")
        raw = read_ref(result["archived_ref"])
    elif source.exists():
        raw = read_ref({"path": str(source), "sha256": expected_sha})
    else:
        raw = read_ref({"path": str(archive), "sha256": expected_sha})
    payload = json.loads(raw)
    body = {"schema_version": payload.get("schema_version"), "intervals": payload.get("intervals")}
    if (
        set(payload) != {"schema_version", "intervals", "record_sha256"}
        or payload["schema_version"] != "daily-basic-coverage-intervals.v2"
        or canonical_json_sha256(body) != payload["record_sha256"]
        or not isinstance(payload["intervals"], list)
        or not payload["intervals"]
        or any(
            not isinstance(row, dict)
            or not re.fullmatch(r"[0-9]{8}", str(row.get("cutoff", "")))
            or row["cutoff"] >= q["effective_date"]
            for row in payload["intervals"]
        )
    ):
        raise RuntimeError("SCOPE_COVERAGE_DECLARATION_NOT_PROVEN_STALE")
    if replayed:
        if (
            set(result)
            != {
                "schema_version",
                "status",
                "request_sha256",
                "source_path",
                "archived_ref",
                "readiness_ref",
                "interval_count",
                "replacement_exemptions_created",
                "reason",
                "recorded_at",
            }
            or result["schema_version"] != "cn-stale-coverage-declaration-retirement.v1"
            or result["status"] != "ARCHIVED"
            or result["source_path"] != str(source)
            or result["interval_count"] != len(payload["intervals"])
            or result["replacement_exemptions_created"] is not False
            or result["reason"] != "PRIOR_CUTOFF_DECLARATION_NOT_VALID_FOR_NEW_PIT_BINDING"
        ):
            raise RuntimeError("SCOPE_COVERAGE_RETIREMENT_RECEIPT_INVALID")
        return {**result, "status": "NO_ACTION", "receipt_ref": reference(receipt)}
    _write_new(archive, raw)
    if source.exists():
        if _bytes(source) != raw:
            raise RuntimeError("SCOPE_COVERAGE_DECLARATION_CHANGED")
        source.unlink()
    result = {
        "schema_version": "cn-stale-coverage-declaration-retirement.v1",
        "status": "ARCHIVED",
        "request_sha256": op.name,
        "source_path": str(source),
        "archived_ref": reference(archive),
        "readiness_ref": reference(op / "readiness.json"),
        "interval_count": len(payload["intervals"]),
        "replacement_exemptions_created": False,
        "reason": "PRIOR_CUTOFF_DECLARATION_NOT_VALID_FOR_NEW_PIT_BINDING",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_new(receipt, encoded(result))
    return {**result, "receipt_ref": reference(receipt)}


def run_scope_transition(
    *,
    workspace_root: str | Path,
    run_root: str | Path,
    mode: str,
    attempt_slot: str,
    request_path: str | Path,
    request_sha256: str,
    retire_coverage_declaration_sha256: str = "",
) -> dict[str, Any]:
    """Receipt-bound recovery branch of daily-maintain; reuses its component DAG."""
    from contextlib import ExitStack
    import fcntl

    from .daily_maintenance import (
        MaintenanceContext,
        _RunLock,
        _attempt_root,
        _seal_attempt,
        _run_component,
    )
    from .daily_components import build_default_components
    from .pit_universe import PITUniverseStore

    q = load_request(request_path, request_sha256)
    root = Path(workspace_root).resolve()
    run = Path(run_root).resolve()
    if (
        mode != "execute"
        or attempt_slot != "2020"
        or str(root) != q["workspace_root"]
        or run != root / "data/private/cn_daily_maintenance"
    ):
        raise RuntimeError("SCOPE_TRANSITION_EXECUTION_SCOPE_INVALID")
    op = run / "scope_transitions" / request_sha256
    marker = marker_path(root)
    with _RunLock(run / ".daily-maintenance.lock"), ExitStack() as stack:
        checkpoint_lock = Path(q["checkpoint_ref"]["path"]).parent / ".checkpoint.lock"
        checkpoint = stack.enter_context(checkpoint_lock.open("rb"))
        fcntl.flock(checkpoint, fcntl.LOCK_EX | fcntl.LOCK_NB)
        read_ref(q["checkpoint_ref"])
        read_ref(q["fundamental_pointer_ref"])
        terminal = op / "terminal.json"
        if terminal.exists():
            result = json.loads(_bytes(terminal))
            with _owned(request_sha256):
                closure = _verify_closure(q)
            if any(
                closure[k] != result["closure"][k] for k in closure if k != "storage_validation"
            ):
                raise RuntimeError("SCOPE_TRANSITION_REPLAY_DRIFT")
            recovery = _recover_exact_veto(q, op, request_sha256)
            ready = _ready_receipt(op)
            with _owned(request_sha256):
                retirement = (
                    _retire_stale_coverage_declaration(q, op, retire_coverage_declaration_sha256)
                    if retire_coverage_declaration_sha256
                    else None
                )
            if marker.exists():
                if json.loads(_bytes(marker))["request_sha256"] != request_sha256:
                    raise RuntimeError("SCOPE_TRANSITION_MARKER_CONFLICT")
                marker.unlink()
            return {**ready, "status": "NO_ACTION", "coverage_declaration_retirement": retirement}
        read_ref(q["veto_ref"])
        if marker.exists() and json.loads(_bytes(marker))["request_sha256"] != request_sha256:
            raise RuntimeError("SCOPE_TRANSITION_ALREADY_RUNNING")
        prepared = op / "prepared.json"
        if not prepared.exists() or not marker.exists():
            read_ref(q["market_pointer_ref"])
            read_ref(q["pit_pointer_ref"])
            if _bytes(q["canonical_scope_path"]) != read_ref(q["old_scope_ref"]):
                raise RuntimeError("SCOPE_TRANSITION_OLD_SCOPE_DRIFT")
            # No in-flight writer may have bound the old inputs before the marker.
            with ExitStack() as writer_locks:
                for p in [
                    root / "data/parquet/cn/.market_writer.lock",
                    root / "data/parquet/cn/reference/.pit_writer.lock",
                ]:
                    h = writer_locks.enter_context(p.open("rb"))
                    fcntl.flock(h, fcntl.LOCK_EX | fcntl.LOCK_NB)
                read_ref(q["market_pointer_ref"])
                read_ref(q["pit_pointer_ref"])
                _write_new(
                    marker,
                    encoded({"request_sha256": request_sha256, "request_path": str(request_path)}),
                )
                _write_new(prepared, encoded({"request_ref": reference(request_path)}))
        _write_new(
            marker, encoded({"request_sha256": request_sha256, "request_path": str(request_path)})
        )
        attempt = _attempt_root(run, now=datetime.now(timezone.utc), slot="2020")
        stage_results: list[dict[str, Any]] = []
        with _owned(request_sha256):
            try:
                scope_raw = _bytes(q["canonical_scope_path"])
                if scope_raw not in (read_ref(q["old_scope_ref"]), read_ref(q["new_scope_ref"])):
                    raise RuntimeError("SCOPE_TRANSITION_SCOPE_DRIFT")
                _replace(Path(q["canonical_scope_path"]), read_ref(q["new_scope_ref"]))
                _write_new(
                    op / "scope-replaced.json", encoded(reference(q["canonical_scope_path"]))
                )
                components = build_default_components(workspace_root=root)
                context = MaintenanceContext(
                    workspace_root=root,
                    run_root=run,
                    attempt_root=attempt,
                    target_date=q["effective_date"],
                    attempt_slot="2020",
                    mode="execute",
                    close_session_receipt=json.loads(read_ref(q["close_receipt_ref"])),
                    close_session_receipt_path=Path(q["close_receipt_ref"]["path"]),
                    close_session_receipt_sha256=q["close_receipt_ref"]["sha256"],
                    scope_transition_request=Path(request_path),
                    expected_scope_transition_sha256=request_sha256,
                )
                for stage, callback in (("PIT", components.pit), ("MARKET", components.market)):
                    result = _run_component(
                        stage=stage, callback=callback, context=context, prior_results=stage_results
                    )
                    stage_results.append(result)
                    if result["status"] not in {"READY", "NO_ACTION"}:
                        raise RuntimeError(f"SCOPE_TRANSITION_{stage}_BLOCKED")
                    pit = PITUniverseStore(
                        root_dir=root / "data/parquet/cn/reference"
                    ).load_generation_binding()
                    if pit["manifest"]["source_bindings"].get("scope_transition") != {
                        "path": str(request_path),
                        "sha256": request_sha256,
                    }:
                        raise RuntimeError("SCOPE_TRANSITION_PIT_COMMIT_DRIFT")
                    commit_file = op / f"{stage.lower()}-committed.json"
                    observed = reference(
                        q["pit_pointer_ref"]["path"]
                        if stage == "PIT"
                        else q["market_pointer_ref"]["path"]
                    )
                    if commit_file.exists() and json.loads(_bytes(commit_file)) != observed:
                        history_receipt = op / "history-repair.json"
                        acquisition = op / "history-acquisition.json"
                        if stage != "MARKET" or not acquisition.exists():
                            raise RuntimeError("SCOPE_TRANSITION_COMMIT_DRIFT")
                        if history_receipt.exists():
                            if (
                                json.loads(_bytes(history_receipt))["after_market_pointer_ref"]
                                != observed
                            ):
                                raise RuntimeError("SCOPE_TRANSITION_HISTORY_COMMIT_DRIFT")
                        else:
                            market = json.loads(_bytes(observed["path"]))
                            manifest = json.loads(_bytes(market["manifest_path"]))
                            if (
                                manifest.get("metadata", {}).get("scope_transition_request_sha256")
                                != request_sha256
                            ):
                                raise RuntimeError("SCOPE_TRANSITION_HISTORY_COMMIT_DRIFT")
                    else:
                        _write_new(commit_file, encoded(observed))
                from .scope_transition_history import repair_and_verify_added_history

                history = repair_and_verify_added_history(q, operation_root=op)
                stage_results.append(
                    {
                        "stage": "SCOPE_HISTORY",
                        "status": "READY",
                        "write_performed": history["repaired_row_count"] > 0,
                        "blockers": [],
                        "evidence": history,
                    }
                )
                result = _run_component(
                    stage="HISTORY",
                    callback=components.history,
                    context=context,
                    prior_results=stage_results[:2],
                )
                stage_results.append(result)
                if result["status"] != "READY":
                    raise RuntimeError("SCOPE_TRANSITION_FULL_HISTORY_BLOCKED")
                closure = _verify_closure(q)
                payload = {
                    "schema_version": "cn-scope-transition-readiness.v1",
                    "status": "DATA_VERIFIED",
                    "request_ref": reference(request_path),
                    "closure": closure,
                    "history": history,
                    "stage_results": stage_results,
                    "claude_input_ready": False,
                    "fundamental_promotion": False,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                sealed = _seal_attempt(attempt_root=attempt, payload=payload, state=payload)
                payload["attempt_receipt_ref"] = sealed["attempt_receipt_ref"]
                _write_new(terminal, encoded(payload))
                recovery = _recover_exact_veto(q, op, request_sha256)
                ready = _ready_receipt(op)
                retirement = (
                    _retire_stale_coverage_declaration(q, op, retire_coverage_declaration_sha256)
                    if retire_coverage_declaration_sha256
                    else None
                )
                marker.unlink()
                return {
                    **ready,
                    "veto_recovery": recovery,
                    "coverage_declaration_retirement": retirement,
                }
            except Exception as exc:
                # Pre-CAS recovery may restore only exact old inputs. After PIT
                # CAS leave the marker and veto; the same request recovers forward.
                if reference(q["pit_pointer_ref"]["path"]) == q["pit_pointer_ref"]:
                    read_ref(q["market_pointer_ref"])
                    _replace(Path(q["canonical_scope_path"]), read_ref(q["old_scope_ref"]))
                    marker.unlink(missing_ok=True)
                payload = {
                    "schema_version": "cn-scope-transition-attempt.v1",
                    "status": "BLOCKED",
                    "request_ref": reference(request_path),
                    "stage_results": stage_results,
                    "blockers": [str(exc)],
                    "error_type": type(exc).__name__,
                    "claude_input_ready": False,
                    "write_veto_ref": q["veto_ref"],
                }
                if (attempt / "attempt.json").exists():
                    attempt = _attempt_root(run, now=datetime.now(timezone.utc), slot="2020")
                return _seal_attempt(attempt_root=attempt, payload=payload, state=payload)
