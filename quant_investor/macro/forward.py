"""Forward-only, observer-only Macro v2 evidence ledger."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Iterator, Mapping
from zoneinfo import ZoneInfo

import pandas as pd

from quant_investor.macro.contracts import canonical_hash, parse_timestamp
from quant_investor.macro.snapshot import build_macro_snapshot
from quant_investor.macro.store import (
    DEFAULT_OBSERVATIONS_ROOT,
    load_observations,
)

FORWARD_EVENT_SCHEMA = "macro-forward-observation.v1"
FORWARD_SUMMARY_SCHEMA = "macro-forward-summary.v1"
FORWARD_MANIFEST_SCHEMA = "macro-forward-generation-manifest.v1"
FORWARD_POINTER_SCHEMA = "macro-forward-pointer.v1"
DEFAULT_FORWARD_ROOT = Path("results/macro_forward_observation")
_OBSERVER_FLAGS = {
    "observer_only": True,
    "production_eligible": False,
    "applied": False,
}
REQUIRED_FORWARD_SESSIONS = 90
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_GENERATION_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_SHANGHAI = ZoneInfo("Asia/Shanghai")
_UTC = ZoneInfo("UTC")


class MacroForwardError(RuntimeError):
    """Raised when forward evidence cannot be proven safe and sequential."""


def _valid_generation_id(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(
        text
        and text not in {".", ".."}
        and _SAFE_GENERATION_ID.fullmatch(text)
    )


def _utc_now() -> datetime:
    return datetime.now(_UTC)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _unsafe_path(path: Path) -> bool:
    current = path.absolute()
    while True:
        if current.exists() and current.is_symlink():
            return True
        if current.parent == current:
            return False
        current = current.parent


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    if path.exists() and path.is_symlink():
        raise MacroForwardError("macro_forward_output_symlink_rejected")
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        _fsync_dir(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    os.fchmod(descriptor, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _calendar(path_value: str | Path) -> tuple[Path, list[str], str]:
    path = Path(path_value).expanduser()
    if (
        not path.exists()
        or _unsafe_path(path)
        or not path.is_file()
        or path.suffix.lower() != ".parquet"
    ):
        raise MacroForwardError("macro_forward_calendar_missing_or_unsafe")
    frame = pd.read_parquet(path)
    if not {"cal_date", "is_open"}.issubset(frame.columns):
        raise MacroForwardError("macro_forward_calendar_schema_invalid")
    dates = pd.to_datetime(frame["cal_date"].astype(str), errors="coerce")
    flags = pd.to_numeric(frame["is_open"], errors="coerce")
    if (
        dates.isna().any()
        or flags.isna().any()
        or not set(flags.unique()).issubset({0.0, 1.0})
    ):
        raise MacroForwardError("macro_forward_calendar_values_invalid")
    normalized = pd.DataFrame(
        {"cal_date": dates.dt.normalize(), "is_open": flags.astype(int)}
    )
    if normalized.groupby("cal_date")["is_open"].nunique().gt(1).any():
        raise MacroForwardError("macro_forward_calendar_date_conflict")
    normalized = normalized.drop_duplicates(subset=["cal_date", "is_open"])
    expected_dates = pd.date_range(
        normalized["cal_date"].min(), normalized["cal_date"].max(), freq="D"
    )
    if set(normalized["cal_date"]) != set(expected_dates):
        raise MacroForwardError("macro_forward_calendar_date_gap")
    calendar_end = normalized["cal_date"].max().date().isoformat()
    open_dates = sorted(
        item.date().isoformat()
        for item in normalized.loc[normalized["is_open"].eq(1), "cal_date"]
    )
    if not open_dates:
        raise MacroForwardError("macro_forward_calendar_no_open_dates")
    return path.resolve(), open_dates, calendar_end


def _target_session(open_dates: list[str], now: datetime) -> str:
    if now.tzinfo is None:
        raise MacroForwardError("macro_forward_clock_timezone_required")
    current = now.astimezone(_SHANGHAI)
    eligible = []
    for value in open_dates:
        day = datetime.fromisoformat(value).date()
        close = datetime.combine(day, time(15, 0), tzinfo=_SHANGHAI)
        if close <= current:
            eligible.append(value)
    if not eligible:
        raise MacroForwardError("macro_forward_no_completed_session")
    return eligible[-1]


def _event_hash(payload: Mapping[str, Any]) -> str:
    stable = dict(payload)
    stable.pop("event_hash", None)
    return canonical_hash(stable)


def _validate_ledger(rows: list[Mapping[str, Any]]) -> None:
    previous_hash = ""
    previous_sequence = 0
    previous_date = ""
    for row in rows:
        if row.get("schema_version") != FORWARD_EVENT_SCHEMA:
            raise MacroForwardError("macro_forward_ledger_schema_invalid")
        sequence = int(row.get("sequence", 0))
        as_of = str(row.get("as_of") or "")
        try:
            as_of_date = date.fromisoformat(as_of)
            session_close = parse_timestamp(
                row.get("session_close"), field_name="session_close"
            )
            recorded_at = parse_timestamp(
                row.get("recorded_at"), field_name="recorded_at"
            )
        except ValueError as exc:
            error = MacroForwardError("macro_forward_ledger_time_invalid")
            raise error from exc
        expected_close = datetime.combine(
            as_of_date, time(15, 0), tzinfo=_SHANGHAI
        ).astimezone(_UTC)
        if session_close != expected_close or recorded_at < session_close:
            raise MacroForwardError("macro_forward_ledger_time_invalid")
        if sequence != previous_sequence + 1 or as_of <= previous_date:
            raise MacroForwardError("macro_forward_ledger_sequence_invalid")
        if str(row.get("previous_event_hash") or "") != previous_hash:
            raise MacroForwardError("macro_forward_ledger_chain_invalid")
        actual_hash = str(row.get("event_hash") or "")
        if not _SHA256.fullmatch(actual_hash) or actual_hash != _event_hash(
            row
        ):
            raise MacroForwardError("macro_forward_ledger_hash_invalid")
        if row.get("observer_only") is not True:
            raise MacroForwardError("macro_forward_ledger_not_observer_only")
        if (
            row.get("production_eligible") is not False
            or row.get("activation_authorized") is not False
            or row.get("applied") is not False
        ):
            raise MacroForwardError(
                "macro_forward_ledger_production_flag_invalid"
            )
        if row.get("market") != "CN" or row.get("readiness_status") not in {
            "pass",
            "degraded",
            "block",
        }:
            raise MacroForwardError("macro_forward_ledger_state_invalid")
        blockers = row.get("blockers")
        selected = row.get("selected_observation_hashes")
        if (
            not isinstance(blockers, list)
            or not all(isinstance(item, str) for item in blockers)
            or not isinstance(selected, list)
            or not all(_SHA256.fullmatch(str(item)) for item in selected)
        ):
            raise MacroForwardError("macro_forward_ledger_evidence_invalid")
        hash_fields = (
            "snapshot_hash",
            "observation_content_set_hash",
            "observation_pointer_sha256",
            "calendar_sha256",
        )
        if not all(
            _SHA256.fullmatch(str(row.get(key) or "")) for key in hash_fields
        ):
            raise MacroForwardError("macro_forward_ledger_evidence_invalid")
        generation_id = str(row.get("observation_generation_id") or "")
        if not _valid_generation_id(generation_id):
            raise MacroForwardError("macro_forward_ledger_evidence_invalid")
        industry_coverage = row.get("industry_chain_coverage")
        try:
            confidence = float(row.get("confidence"))
            national_coverage = float(row.get("national_coverage"))
            industry_values = [
                float(value)
                for value in (
                    industry_coverage.values()
                    if isinstance(industry_coverage, Mapping)
                    else []
                )
            ]
        except (TypeError, ValueError) as exc:
            error = MacroForwardError("macro_forward_ledger_metric_invalid")
            raise error from exc
        if (
            not math.isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
            or not math.isfinite(national_coverage)
            or not 0.0 <= national_coverage <= 1.0
            or not isinstance(industry_coverage, Mapping)
            or not all(
                math.isfinite(value) and 0.0 <= value <= 1.0
                for value in industry_values
            )
        ):
            raise MacroForwardError("macro_forward_ledger_metric_invalid")
        previous_hash = actual_hash
        previous_sequence = sequence
        previous_date = as_of


def _validate_sessions_against_calendar(
    rows: list[Mapping[str, Any]], open_dates: list[str]
) -> None:
    positions = {value: index for index, value in enumerate(open_dates)}
    indices: list[int] = []
    for row in rows:
        as_of = str(row.get("as_of") or "")
        if as_of not in positions:
            raise MacroForwardError("macro_forward_ledger_session_not_open")
        indices.append(positions[as_of])
    if any(right != left + 1 for left, right in zip(indices, indices[1:])):
        raise MacroForwardError("macro_forward_ledger_session_gap")


def _load_generation(
    market_root: Path,
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any], str]:
    pointer_path = market_root / "_latest.json"
    if not pointer_path.exists():
        return [], {}, ""
    if _unsafe_path(pointer_path) or not pointer_path.is_file():
        raise MacroForwardError("macro_forward_pointer_unsafe")
    pointer_bytes = pointer_path.read_bytes()
    try:
        pointer = json.loads(pointer_bytes)
    except Exception as exc:
        raise MacroForwardError("macro_forward_pointer_invalid") from exc
    if (
        not isinstance(pointer, Mapping)
        or pointer.get("schema_version") != FORWARD_POINTER_SCHEMA
    ):
        raise MacroForwardError("macro_forward_pointer_schema_invalid")
    if any(
        pointer.get(key) is not value
        for key, value in _OBSERVER_FLAGS.items()
    ):
        raise MacroForwardError(
            "macro_forward_pointer_observer_flags_invalid"
        )
    generation_id = str(pointer.get("generation_id") or "")
    if not _valid_generation_id(generation_id):
        raise MacroForwardError("macro_forward_generation_id_invalid")
    generation = market_root / "_generations" / generation_id
    manifest_path = generation / "manifest.json"
    ledger_path = generation / "ledger.jsonl"
    summary_path = generation / "summary.json"
    for artifact in (generation, manifest_path, ledger_path, summary_path):
        if not artifact.exists() or _unsafe_path(artifact):
            raise MacroForwardError("macro_forward_generation_unsafe")
    if generation.stat().st_mode & 0o077:
        raise MacroForwardError("macro_forward_generation_mode_unsafe")
    protected_files = (pointer_path, manifest_path, ledger_path, summary_path)
    if any(path.stat().st_mode & 0o077 for path in protected_files):
        raise MacroForwardError("macro_forward_artifact_mode_unsafe")
    manifest_bytes = manifest_path.read_bytes()
    if _sha256(manifest_bytes) != str(pointer.get("manifest_sha256") or ""):
        raise MacroForwardError("macro_forward_manifest_hash_mismatch")
    manifest = json.loads(manifest_bytes)
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema_version") != FORWARD_MANIFEST_SCHEMA
        or manifest.get("generation_id") != generation_id
    ):
        raise MacroForwardError("macro_forward_manifest_schema_invalid")
    if any(
        manifest.get(key) is not value
        for key, value in _OBSERVER_FLAGS.items()
    ):
        raise MacroForwardError(
            "macro_forward_manifest_observer_flags_invalid"
        )
    ledger_bytes = ledger_path.read_bytes()
    summary_bytes = summary_path.read_bytes()
    if _sha256(ledger_bytes) != manifest.get("ledger_sha256"):
        raise MacroForwardError("macro_forward_ledger_file_hash_mismatch")
    if _sha256(summary_bytes) != manifest.get("summary_sha256"):
        raise MacroForwardError("macro_forward_summary_hash_mismatch")
    rows: list[Mapping[str, Any]] = []
    try:
        for line in ledger_bytes.decode("utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise TypeError("row_not_mapping")
                rows.append(dict(row))
    except Exception as exc:
        raise MacroForwardError("macro_forward_ledger_invalid") from exc
    _validate_ledger(rows)
    if len(rows) != int(manifest.get("event_count", -1)):
        raise MacroForwardError("macro_forward_event_count_mismatch")
    try:
        persisted_summary = json.loads(summary_bytes)
    except Exception as exc:
        raise MacroForwardError("macro_forward_summary_invalid") from exc
    if not rows or persisted_summary != _summary(rows):
        raise MacroForwardError("macro_forward_summary_readback_mismatch")
    if (
        manifest.get("latest_event_hash") != rows[-1]["event_hash"]
        or pointer.get("latest_event_hash") != rows[-1]["event_hash"]
        or int(pointer.get("event_count", -1)) != len(rows)
    ):
        raise MacroForwardError("macro_forward_generation_metadata_mismatch")
    return rows, pointer, _sha256(pointer_bytes)


def forward_pointer_sha256(
    root: str | Path = DEFAULT_FORWARD_ROOT,
    *,
    market: str = "CN",
) -> str:
    path = Path(root).expanduser() / str(market).upper() / "_latest.json"
    if not path.exists():
        return ""
    if _unsafe_path(path) or not path.is_file():
        raise MacroForwardError("macro_forward_pointer_unsafe")
    return _sha256(path.read_bytes())


def _summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    ready_count = sum(row.get("readiness_status") == "pass" for row in rows)
    last = rows[-1]
    observed = len(rows)
    duration_reached = observed >= REQUIRED_FORWARD_SESSIONS
    maturity_blockers = ["outcome_stability_evidence_not_implemented"]
    if not duration_reached:
        maturity_blockers.append("forward_sessions_below_90")
    if ready_count != observed:
        maturity_blockers.append("readiness_gaps_present")
    return {
        "schema_version": FORWARD_SUMMARY_SCHEMA,
        "status": "observing",
        "market": "CN",
        "first_session": rows[0]["as_of"],
        "latest_session": last["as_of"],
        "observed_forward_sessions": observed,
        "ready_sessions": ready_count,
        "degraded_or_blocked_sessions": observed - ready_count,
        "required_forward_sessions": REQUIRED_FORWARD_SESSIONS,
        "remaining_forward_sessions": max(
            REQUIRED_FORWARD_SESSIONS - observed, 0
        ),
        "forward_duration_reached": duration_reached,
        "measurement_maturity_reached": False,
        "maturity_blockers": sorted(maturity_blockers),
        "latest_snapshot_hash": last["snapshot_hash"],
        "latest_event_hash": last["event_hash"],
        "observer_only": True,
        "production_eligible": False,
        "activation_authorized": False,
        "applied": False,
    }


def record_macro_forward_observation(
    *,
    market: str = "CN",
    observations_root: str | Path = DEFAULT_OBSERVATIONS_ROOT,
    calendar_path: str | Path,
    root: str | Path = DEFAULT_FORWARD_ROOT,
    expected_pointer_sha256: str,
) -> dict[str, Any]:
    """Record exactly one newly completed trading session; never backfill."""

    if str(market).upper() != "CN":
        raise MacroForwardError("macro_forward_market_unsupported")
    if expected_pointer_sha256 is None:
        raise MacroForwardError("macro_forward_expected_pointer_required")
    expected_pointer = str(expected_pointer_sha256).lower()
    if expected_pointer and not _SHA256.fullmatch(expected_pointer):
        raise MacroForwardError("macro_forward_expected_pointer_invalid")
    calendar, open_dates, calendar_end = _calendar(calendar_path)
    now = _utc_now()
    if calendar_end < now.astimezone(_SHANGHAI).date().isoformat():
        raise MacroForwardError("macro_forward_calendar_stale")
    target = _target_session(open_dates, now)
    rows, observation_generation = load_observations(observations_root)
    generation_id = str(observation_generation.get("generation_id") or "")
    generation_hash = str(observation_generation.get("content_set_hash") or "")
    pointer_hash = str(observation_generation.get("pointer_sha256") or "")
    if (
        not rows
        or not generation_id
        or not generation_hash
        or not pointer_hash
    ):
        raise MacroForwardError("macro_forward_observation_generation_missing")
    snapshot = build_macro_snapshot(rows, market="CN", as_of=target)

    state_root = Path(root).expanduser()
    if _unsafe_path(state_root):
        raise MacroForwardError("macro_forward_root_symlink_rejected")
    state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(state_root, 0o700)
    market_root = state_root / "CN"
    if _unsafe_path(market_root):
        raise MacroForwardError("macro_forward_market_root_symlink_rejected")
    market_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(market_root, 0o700)
    generations = market_root / "_generations"
    generations.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(generations, 0o700)

    with _locked(market_root / ".forward.lock"):
        existing, previous_pointer, current_pointer_hash = _load_generation(
            market_root
        )
        if current_pointer_hash != expected_pointer:
            raise MacroForwardError("macro_forward_pointer_cas_mismatch")
        _validate_sessions_against_calendar(existing, open_dates)
        if existing:
            previous = existing[-1]
            previous_date = str(previous["as_of"])
            if target == previous_date:
                if (
                    snapshot.snapshot_hash != previous["snapshot_hash"]
                    or generation_id != previous["observation_generation_id"]
                    or _sha256(calendar.read_bytes())
                    != previous["calendar_sha256"]
                ):
                    raise MacroForwardError("macro_forward_same_session_drift")
                return {
                    **_summary(existing),
                    "promoted": False,
                    "idempotent": True,
                    "pointer_sha256": current_pointer_hash,
                }
            subsequent = [
                item for item in open_dates if previous_date < item <= target
            ]
            if subsequent != [target]:
                raise MacroForwardError("macro_forward_session_gap_detected")
        sequence = len(existing) + 1
        event: dict[str, Any] = {
            "schema_version": FORWARD_EVENT_SCHEMA,
            "sequence": sequence,
            "previous_event_hash": (
                str(existing[-1]["event_hash"]) if existing else ""
            ),
            "market": "CN",
            "as_of": target,
            "session_close": snapshot.published_cutoff,
            "recorded_at": now.astimezone(_UTC).isoformat(),
            "snapshot_hash": snapshot.snapshot_hash,
            "readiness_status": snapshot.readiness_status,
            "blockers": list(snapshot.blockers),
            "national_coverage": snapshot.coverage.get("national"),
            "industry_chain_coverage": snapshot.coverage.get(
                "industry_chains"
            ),
            "confidence": snapshot.confidence,
            "selected_observation_hashes": list(
                snapshot.selected_observation_hashes
            ),
            "observation_generation_id": generation_id,
            "observation_content_set_hash": generation_hash,
            "observation_pointer_sha256": pointer_hash,
            "calendar_sha256": _sha256(calendar.read_bytes()),
            "observer_only": True,
            "production_eligible": False,
            "activation_authorized": False,
            "applied": False,
        }
        event["event_hash"] = _event_hash(event)
        updated = [*existing, event]
        _validate_ledger(updated)
        _validate_sessions_against_calendar(updated, open_dates)
        summary = _summary(updated)
        ledger_bytes = b"".join(
            _canonical_bytes(row) + b"\n" for row in updated
        )
        summary_bytes = json.dumps(
            summary, ensure_ascii=False, sort_keys=True, indent=2
        ).encode("utf-8")
        generation_name = f"{sequence:04d}_{target}_{event['event_hash'][:12]}"
        final = generations / generation_name
        if final.exists():
            raise MacroForwardError("macro_forward_generation_exists")
        staging = Path(
            tempfile.mkdtemp(prefix=f".{generation_name}.", dir=generations)
        )
        os.chmod(staging, 0o700)
        pointer_advanced = False
        try:
            _atomic_bytes(staging / "ledger.jsonl", ledger_bytes)
            _atomic_bytes(staging / "summary.json", summary_bytes)
            manifest = {
                "schema_version": FORWARD_MANIFEST_SCHEMA,
                "generation_id": generation_name,
                "parent_generation_id": previous_pointer.get("generation_id"),
                "event_count": len(updated),
                "latest_event_hash": event["event_hash"],
                "ledger_sha256": _sha256(ledger_bytes),
                "summary_sha256": _sha256(summary_bytes),
                "observer_only": True,
                "production_eligible": False,
                "applied": False,
            }
            manifest_bytes = json.dumps(
                manifest, ensure_ascii=False, sort_keys=True, indent=2
            ).encode("utf-8")
            _atomic_bytes(staging / "manifest.json", manifest_bytes)
            _fsync_dir(staging)
            os.replace(staging, final)
            _fsync_dir(generations)
            pointer = {
                "schema_version": FORWARD_POINTER_SCHEMA,
                "generation_id": generation_name,
                "manifest_sha256": _sha256(manifest_bytes),
                "latest_event_hash": event["event_hash"],
                "event_count": len(updated),
                **_OBSERVER_FLAGS,
            }
            pointer_bytes = _canonical_bytes(pointer)
            _atomic_bytes(market_root / "_latest.json", pointer_bytes)
            pointer_advanced = True
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            if final.exists() and not pointer_advanced:
                keep_final = False
                pointer_path = market_root / "_latest.json"
                if pointer_path.exists() and not pointer_path.is_symlink():
                    try:
                        current = json.loads(pointer_path.read_bytes())
                        keep_final = (
                            isinstance(current, Mapping)
                            and current.get("generation_id") == generation_name
                        )
                    except Exception:
                        keep_final = False
                if not keep_final:
                    shutil.rmtree(final)
            raise
        return {
            **summary,
            "promoted": True,
            "idempotent": False,
            "generation_id": generation_name,
            "pointer_sha256": _sha256(pointer_bytes),
        }


__all__ = [
    "DEFAULT_FORWARD_ROOT",
    "FORWARD_EVENT_SCHEMA",
    "MacroForwardError",
    "forward_pointer_sha256",
    "record_macro_forward_observation",
]
