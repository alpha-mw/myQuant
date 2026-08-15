#!/usr/bin/env python3
"""Append or inspect the local, bounded advisory governance log.

``decision_log.v1`` remains readable for compatibility.  New weekly formal
advice uses one non-executable ``decision_log.v2`` advisory envelope.  The
writer is append-only, lock-serialized, exact-readback verified, and has no
provider, broker, order, execution, or trade capability.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_PATH = PROJECT_ROOT / "results" / "decision_log" / "decision_log.jsonl"
EVENT_TYPES = {"pipeline_proposal", "advisory", "human_action", "advisory_envelope"}
ADVISORY_SOURCES = {"codex_thread", "claude", "workbench", "other"}
FORMAL_ACTIONS = {"BUY", "ADD", "REDUCE", "EXIT", "HOLD", "WATCH"}
FORMAL_OUTCOMES = {"ADVISORY", "NO_ACTION"}
MAX_ACTIONS = 50
MAX_LINE_BYTES = 128 * 1024
MAX_LOG_BYTES = 32 * 1024 * 1024
MAX_TEXT = 4000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REPORT_WEEK = re.compile(r"^[0-9]{4}-W[0-9]{2}$")
_SYMBOL = re.compile(r"^[0-9]{6}\.(?:SH|SZ|BJ)$")


class DecisionLogError(RuntimeError):
    """Fail-closed decision-log storage or contract error."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _canonical_payload_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DecisionLogError("decision log contains non-canonical JSON data") from exc


def _canonical_line(value: Any) -> bytes:
    return _canonical_payload_bytes(value) + b"\n"


def _semantic_sha(value: dict[str, Any]) -> str:
    body = {
        key: value[key]
        for key in sorted(value)
        if key not in {"event_id", "recorded_at", "semantic_sha256"}
    }
    return hashlib.sha256(_canonical_payload_bytes(body)).hexdigest()


def _require_text(value: Any, *, label: str, max_length: int = MAX_TEXT) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > max_length
        or "\x00" in value
        or "\n" in value
        or "\r" in value
    ):
        raise DecisionLogError(f"{label} is not bounded canonical text")
    return value


def _require_sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise DecisionLogError(f"{label} is not a canonical SHA-256")
    return value


def _require_timestamp(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise DecisionLogError(f"{label} is invalid")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise DecisionLogError(f"{label} is not a canonical UTC timestamp") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise DecisionLogError(f"{label} is not canonical")
    return value


def _load_metadata(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("--metadata-json must decode to an object")
    return payload


def _action_id(action: dict[str, Any]) -> str:
    body = {key: action[key] for key in sorted(action) if key != "action_id"}
    return "action-" + hashlib.sha256(_canonical_payload_bytes(body)).hexdigest()[:24]


def _validate_action(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise DecisionLogError("advisory action is not an object")
    action = dict(raw)
    symbol = action.get("symbol")
    if not isinstance(symbol, str) or _SYMBOL.fullmatch(symbol) is None:
        raise DecisionLogError("advisory action symbol is invalid")
    _require_text(action.get("company_name"), label="action company_name", max_length=200)
    direction = action.get("action")
    if direction not in FORMAL_ACTIONS:
        raise DecisionLogError("advisory action direction is invalid")
    shares = action.get("shares_delta")
    if not isinstance(shares, int) or isinstance(shares, bool):
        raise DecisionLogError("advisory action shares_delta must be an integer")
    if direction in {"BUY", "ADD"} and shares <= 0:
        raise DecisionLogError("BUY/ADD shares_delta must be positive")
    if direction in {"REDUCE", "EXIT"} and shares >= 0:
        raise DecisionLogError("REDUCE/EXIT shares_delta must be negative")
    if direction in {"HOLD", "WATCH"} and shares != 0:
        raise DecisionLogError("HOLD/WATCH shares_delta must be zero")
    _require_text(action.get("validity"), label="action validity")
    _require_text(action.get("invalidation"), label="action invalidation")
    refs = action.get("evidence_refs")
    if not isinstance(refs, list) or not refs:
        raise DecisionLogError("advisory action evidence_refs are absent")
    for ref in refs:
        if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
            raise DecisionLogError("advisory action evidence ref shape is invalid")
        _require_text(ref.get("path"), label="action evidence path", max_length=1000)
        _require_sha(ref.get("sha256"), label="action evidence SHA")
    expected_id = _action_id(action)
    supplied = action.get("action_id")
    if supplied is not None and supplied != expected_id:
        raise DecisionLogError("advisory action_id mismatch")
    action["action_id"] = expected_id
    return action


def _make_envelope(payload: dict[str, Any]) -> dict[str, Any]:
    event = dict(payload)
    event["schema_version"] = "decision_log.v2"
    event["event_type"] = "advisory_envelope"
    _require_text(event.get("report_group_id"), label="report_group_id", max_length=200)
    idempotency_key = _require_text(
        event.get("idempotency_key"), label="idempotency_key", max_length=300
    )
    report_week = event.get("report_week")
    if not isinstance(report_week, str) or _REPORT_WEEK.fullmatch(report_week) is None:
        raise DecisionLogError("report_week is invalid")
    _require_timestamp(event.get("scheduled_at"), label="scheduled_at")
    if event.get("canonical_strategy_id") != "cn-aggressive-tech-manufacturing":
        raise DecisionLogError("canonical strategy ID is invalid")
    for field in (
        "identity_sha256",
        "v17_active_run_sha256",
        "v17_active_pointer_sha256",
        "store_pointer_sha256",
        "catalog_sha256",
        "performance_manifest_sha256",
        "financial_state_sha256",
    ):
        _require_sha(event.get(field), label=field)
    if event.get("executable") is not False:
        raise DecisionLogError("advisory envelope must be non-executable")
    if event.get("formal_outcome") not in FORMAL_OUTCOMES:
        raise DecisionLogError("formal_outcome is invalid")
    actions_raw = event.get("actions")
    if not isinstance(actions_raw, list) or len(actions_raw) > MAX_ACTIONS:
        raise DecisionLogError("advisory envelope actions exceed the bound")
    actions = [_validate_action(row) for row in actions_raw]
    actions.sort(key=lambda row: (row["symbol"], row["action"], row["action_id"]))
    if len({row["action_id"] for row in actions}) != len(actions):
        raise DecisionLogError("advisory envelope action_id is duplicated")
    if event["formal_outcome"] == "NO_ACTION" and actions:
        raise DecisionLogError("NO_ACTION envelope cannot contain actions")
    if event["formal_outcome"] == "ADVISORY" and not actions:
        raise DecisionLogError("ADVISORY envelope requires actions")
    event["actions"] = actions
    supersedes = event.get("supersedes_event_id")
    if supersedes is not None:
        _require_text(supersedes, label="supersedes_event_id", max_length=200)
    event["recorded_at"] = event.get("recorded_at") or _utc_now()
    _require_timestamp(event["recorded_at"], label="recorded_at")
    semantic = _semantic_sha(event)
    supplied_semantic = event.get("semantic_sha256")
    if supplied_semantic is not None and supplied_semantic != semantic:
        raise DecisionLogError("advisory envelope semantic_sha256 mismatch")
    event["semantic_sha256"] = semantic
    expected_id = "decision-" + hashlib.sha256(
        (idempotency_key + ":" + semantic).encode("utf-8")
    ).hexdigest()[:32]
    supplied_id = event.get("event_id")
    if supplied_id is not None and supplied_id != expected_id:
        raise DecisionLogError("advisory envelope event_id mismatch")
    event["event_id"] = expected_id
    line = _canonical_line(event)
    if len(line) > MAX_LINE_BYTES:
        raise DecisionLogError("advisory envelope exceeds 128 KiB")
    return event


def make_event(payload: dict[str, Any]) -> dict[str, Any]:
    event_type = str(payload.get("event_type") or "").strip()
    if event_type == "advisory_envelope" or payload.get("schema_version") == "decision_log.v2":
        return _make_envelope(payload)
    event = dict(payload)
    if event_type not in EVENT_TYPES - {"advisory_envelope"}:
        raise ValueError(f"event_type must be one of {sorted(EVENT_TYPES)}")
    event.setdefault("schema_version", "decision_log.v1")
    if event["schema_version"] != "decision_log.v1":
        raise DecisionLogError("legacy decision-log schema is invalid")
    event.setdefault("recorded_at", _utc_now())
    event.setdefault("metadata", {})
    if event_type == "advisory":
        source = str(event.get("answer_source") or "").strip()
        if source not in ADVISORY_SOURCES:
            raise ValueError(f"advisory answer_source must be one of {sorted(ADVISORY_SOURCES)}")
        if not str(event.get("question") or "").strip():
            raise ValueError("advisory requires question")
        if not str(event.get("answer_summary") or "").strip():
            raise ValueError("advisory requires answer_summary")
    if event_type == "human_action" and not str(event.get("action") or "").strip():
        raise ValueError("human_action requires action")
    if event_type == "pipeline_proposal" and not str(
        event.get("proposal_summary") or ""
    ).strip():
        raise ValueError("pipeline_proposal requires proposal_summary")
    event["event_id"] = event.get("event_id") or _event_id(event)
    if len(_canonical_line(event)) > MAX_LINE_BYTES:
        raise DecisionLogError("decision-log event exceeds 128 KiB")
    return event


def _event_id(event: dict[str, Any]) -> str:
    basis = {
        key: event.get(key)
        for key in sorted(event)
        if key not in {"event_id", "recorded_at"}
    }
    return "decision-" + hashlib.sha256(_canonical_payload_bytes(basis)).hexdigest()[:16]


def _validate_parent(path: Path) -> None:
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    metadata = os.lstat(parent)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        raise DecisionLogError("decision-log parent directory is unsafe")


def _validate_open_file(descriptor: int, *, label: str) -> os.stat_result:
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise DecisionLogError(f"{label} must be owner-owned regular single-link mode 0600")
    return metadata


def _read_log_bytes(path: Path) -> bytes:
    if not path.exists():
        return b""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise DecisionLogError("decision log is unavailable or unsafe") from exc
    try:
        before = _validate_open_file(descriptor, label="decision log")
        if before.st_size > MAX_LOG_BYTES:
            raise DecisionLogError("decision log exceeds byte budget")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        identity = lambda item: (  # noqa: E731
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_nlink,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )
        if remaining or identity(before) != identity(after):
            raise DecisionLogError("decision log changed during read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _parse_events(raw: bytes) -> list[dict[str, Any]]:
    if not raw:
        return []
    if not raw.endswith(b"\n"):
        raise DecisionLogError("DECISION_LOG_CORRUPT_TAIL")
    rows: list[dict[str, Any]] = []
    event_ids: set[str] = set()
    idempotency_keys: set[str] = set()
    for line_no, line in enumerate(raw.splitlines(keepends=True), start=1):
        if not line.endswith(b"\n") or len(line) > MAX_LINE_BYTES or line == b"\n":
            raise DecisionLogError(f"decision log row {line_no} is malformed")
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DecisionLogError(f"decision log row {line_no} is malformed") from exc
        if not isinstance(value, dict):
            raise DecisionLogError(f"decision log row {line_no} is not an object")
        event_id = value.get("event_id")
        if not isinstance(event_id, str) or not event_id or event_id in event_ids:
            raise DecisionLogError("decision log event_id is absent or duplicated")
        if value.get("schema_version") == "decision_log.v2":
            if _canonical_line(value) != line:
                raise DecisionLogError("stored advisory envelope is not canonical JSON")
            validated = _make_envelope(value)
            if validated != value:
                raise DecisionLogError("stored advisory envelope is not canonical")
            key = value["idempotency_key"]
            if key in idempotency_keys:
                raise DecisionLogError("decision log idempotency key is duplicated")
            idempotency_keys.add(key)
        elif value.get("schema_version") != "decision_log.v1":
            raise DecisionLogError("decision log schema is unsupported")
        event_ids.add(event_id)
        rows.append(value)
    return rows


def append_event(path: Path, event: dict[str, Any]) -> dict[str, Any]:
    resolved = make_event(event)
    _validate_parent(path)
    lock_path = path.with_name(path.name + ".lock")
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    lock_fd = os.open(lock_path, flags, 0o600)
    try:
        _validate_open_file(lock_fd, label="decision-log lock")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        existing_raw = _read_log_bytes(path)
        existing = _parse_events(existing_raw)
        if resolved.get("schema_version") == "decision_log.v2":
            same_key = [
                row
                for row in existing
                if row.get("schema_version") == "decision_log.v2"
                and row.get("idempotency_key") == resolved["idempotency_key"]
            ]
            if same_key:
                if same_key[0].get("semantic_sha256") == resolved["semantic_sha256"]:
                    return {**same_key[0], "already_recorded": True}
                raise DecisionLogError("DECISION_LOG_IDEMPOTENCY_CONFLICT")
            supersedes = resolved.get("supersedes_event_id")
            same_week = [
                row
                for row in existing
                if row.get("schema_version") == "decision_log.v2"
                and row.get("report_week") == resolved["report_week"]
                and row.get("canonical_strategy_id")
                == resolved["canonical_strategy_id"]
            ]
            if same_week:
                if supersedes != same_week[-1].get("event_id"):
                    raise DecisionLogError(
                        "same-week active V17 update requires explicit supersession"
                    )
            elif supersedes is not None:
                raise DecisionLogError(
                    "supersedes_event_id has no same-week predecessor"
                )
        raw_line = _canonical_line(resolved)
        log_flags = (
            os.O_WRONLY
            | os.O_APPEND
            | os.O_CREAT
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        log_fd = os.open(path, log_flags, 0o600)
        try:
            before = _validate_open_file(log_fd, label="decision log")
            if before.st_size + len(raw_line) > MAX_LOG_BYTES:
                raise DecisionLogError("decision log append exceeds byte budget")
            written = os.write(log_fd, raw_line)
            if written != len(raw_line):
                raise DecisionLogError("decision log append was partial")
            os.fsync(log_fd)
        finally:
            os.close(log_fd)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        readback_raw = _read_log_bytes(path)
        readback = _parse_events(readback_raw)
        if not readback or readback[-1] != resolved:
            raise DecisionLogError("decision log exact readback mismatch")
        if resolved.get("schema_version") == "decision_log.v2":
            return {**resolved, "already_recorded": False}
        return resolved
    finally:
        os.close(lock_fd)


def read_events(path: Path) -> list[dict[str, Any]]:
    return _parse_events(_read_log_bytes(path))


def harden_log_permissions(path: Path) -> dict[str, Any]:
    _validate_parent(path)
    metadata = os.lstat(path)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.getuid()
    ):
        raise DecisionLogError("decision log cannot be safely hardened")
    before = path.read_bytes()
    before_sha = hashlib.sha256(before).hexdigest()
    os.chmod(path, 0o600, follow_symlinks=False)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
        _validate_open_file(descriptor, label="decision log")
    finally:
        os.close(descriptor)
    after = path.read_bytes()
    after_sha = hashlib.sha256(after).hexdigest()
    if before != after or before_sha != after_sha:
        raise DecisionLogError("decision log content changed while hardening permissions")
    _parse_events(after)
    return {
        "hardened": True,
        "mode_before": oct(stat.S_IMODE(metadata.st_mode)),
        "mode_after": "0o600",
        "content_sha256_before": before_sha,
        "content_sha256_after": after_sha,
        "content_unchanged": True,
    }


def build_event_from_args(args: argparse.Namespace) -> dict[str, Any]:
    event = {
        "event_type": args.event_type,
        "trade_date": args.trade_date,
        "channel": args.channel,
        "question": args.question,
        "answer_summary": args.answer_summary,
        "answer_source": args.answer_source,
        "symbol": args.symbol,
        "action": args.action,
        "proposal_summary": args.proposal_summary,
        "rejected_options": [item for item in (args.rejected_option or []) if item],
        "regime_state": args.regime_state,
        "machine_suggestion": args.machine_suggestion,
        "metadata": _load_metadata(args.metadata_json),
    }
    return {key: value for key, value in event.items() if value not in (None, "", [])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--list", type=int, default=0)
    parser.add_argument("--harden-permissions", action="store_true")
    parser.add_argument("--envelope-json", type=Path)
    parser.add_argument("--event-type", choices=sorted(EVENT_TYPES))
    parser.add_argument("--trade-date")
    parser.add_argument("--channel")
    parser.add_argument("--question")
    parser.add_argument("--answer-summary")
    parser.add_argument("--answer-source", choices=sorted(ADVISORY_SOURCES))
    parser.add_argument("--symbol")
    parser.add_argument("--action")
    parser.add_argument("--proposal-summary")
    parser.add_argument("--rejected-option", action="append")
    parser.add_argument("--regime-state")
    parser.add_argument("--machine-suggestion")
    parser.add_argument("--metadata-json")
    args = parser.parse_args()
    try:
        if args.harden_permissions:
            result = harden_log_permissions(args.log_path)
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return
        if args.list:
            events = read_events(args.log_path)
            print(json.dumps(events[-args.list:], ensure_ascii=False, indent=2, sort_keys=True))
            return
        if args.envelope_json is not None:
            payload = json.loads(args.envelope_json.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise DecisionLogError("--envelope-json must contain an object")
            event = append_event(args.log_path, payload)
        else:
            if not args.event_type:
                raise DecisionLogError(
                    "--event-type is required unless --list or --envelope-json is used"
                )
            event = append_event(args.log_path, build_event_from_args(args))
        print(json.dumps(event, ensure_ascii=False, indent=2, sort_keys=True))
    except (DecisionLogError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
