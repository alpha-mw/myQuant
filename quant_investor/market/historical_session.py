"""Historical Calendar evidence derived from original bytes, never a write permit."""

import hashlib
from pathlib import Path
import stat

from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from .close_session_authority import replay_close_session_authority
from .requested_session import classify_catchup_session

SCHEMA = "cn-historical-close-session.v1"
FILENAME = "historical-session.v1.json"
CALENDAR_FILENAME = "close-session-receipt.json"
RAW_FILENAME = "close-session.raw.json"
CORE_SCHEMA = "cn-daily-maintenance-core.v2"
CORE_FIELDS = frozenset(
    {
        "schema_version",
        "logical_claim_ref",
        "producer",
        "scope",
        "other_authority",
        "mode",
        "target_date",
        "status",
        "maintenance_status",
        "started_ref",
        "stage_refs",
        "stage_results",
        "blockers",
        "provider_activity",
        "close_session_receipt_ref",
        "historical_session_ref",
        "sealed_at",
    }
)


class HistoricalSessionError(ValueError):
    """A controlled historical fileset or source-binding failure."""


def prepare_historical_maintenance_input(*, workspace, value, now) -> dict:
    """Snapshot code-owned Calendar inputs before any maintenance journal write."""
    from quant_investor.system.storage import SecureSystemStorage
    from quant_investor.operations.daily_contract import utc_stamp, validate_ref
    from zoneinfo import ZoneInfo

    if type(value) is not dict or set(value) != {
        "target_trade_date",
        "previous_trade_date",
        "calendar_ref",
        "raw_calendar_ref",
    }:
        raise HistoricalSessionError("historical_maintenance_input_fields_invalid")
    value = parse_canonical_json_bytes(canonical_json_bytes(value))
    storage = SecureSystemStorage(str(workspace))
    saved = {}
    for key in ("calendar_ref", "raw_calendar_ref"):
        ref = validate_ref(value[key])
        stored = storage.read_workspace_file_bytes(ref["path"], maximum_bytes=4 * 1024 * 1024)
        if stored.byte_sha256 != ref["sha256"]:
            raise HistoricalSessionError("historical_maintenance_input_sha_mismatch")
        saved[ref["path"]] = stored.data
    calendar = parse_canonical_json_bytes(saved[value["calendar_ref"]["path"]])
    raw = saved[value["raw_calendar_ref"]["path"]]
    projection = classify_catchup_session(
        requested_trade_date=value["target_trade_date"],
        previous_trade_date=value["previous_trade_date"],
        receipt=calendar,
        raw=raw,
    )
    if utc_stamp(projection["observed_at"]) > now or value["target_trade_date"] >= now.astimezone(
        ZoneInfo("Asia/Shanghai")
    ).strftime("%Y%m%d"):
        raise HistoricalSessionError("historical_maintenance_input_time_invalid")
    verified = replay_close_session_authority(calendar, raw)

    def recheck():
        for path, expected in saved.items():
            if (
                storage.read_workspace_file_bytes(path, maximum_bytes=4 * 1024 * 1024).data
                != expected
            ):
                raise HistoricalSessionError("historical_maintenance_input_changed")

    recheck()
    return {
        "target_trade_date": value["target_trade_date"],
        "previous_trade_date": value["previous_trade_date"],
        "calendar": verified.receipt,
        "raw": verified.raw_response_bytes,
        "recheck": recheck,
    }


def read_historical_fileset(*, path, proof_raw: bytes, read) -> dict:
    """Read-only custody; the caller supplies its own stable, safe byte reader."""
    path = Path(path).expanduser().absolute()
    parent = path.parent
    if path.name != FILENAME or parent.resolve(strict=True) != parent:
        raise HistoricalSessionError("historical_authority_path_invalid")
    before = parent.stat()
    if not stat.S_ISDIR(before.st_mode):
        raise HistoricalSessionError("historical_authority_parent_invalid")
    calendar_path, raw_path = parent / CALENDAR_FILENAME, parent / RAW_FILENAME
    calendar_bytes = read(calendar_path, label="historical_calendar")
    raw = read(raw_path, label="historical_calendar_raw")
    proof = verify_historical_session(proof_bytes=proof_raw, calendar_bytes=calendar_bytes, raw=raw)
    for selected, expected in ((path, proof_raw), (calendar_path, calendar_bytes), (raw_path, raw)):
        if read(selected, label="historical_authority_recheck") != expected:
            raise HistoricalSessionError("historical_authority_changed")
    after = parent.stat()
    if parent.resolve(strict=True) != parent or (before.st_dev, before.st_ino) != (
        after.st_dev,
        after.st_ino,
    ):
        raise HistoricalSessionError("historical_authority_parent_changed")
    return proof


def read_historical_core(*, attempt_root, proof_ref, close_ref, started_ref, target, read) -> dict:
    """Bind one historical checkpoint to original Calendar and actual start bytes."""
    from quant_investor.operations.daily_contract import utc_stamp

    parent = Path(attempt_root).absolute()
    before = parent.stat()
    saved = {}
    for ref, name in (
        (proof_ref, FILENAME),
        (close_ref, CALENDAR_FILENAME),
        (started_ref, "started.json"),
    ):
        if (
            type(ref) is not dict
            or set(ref) != {"path", "sha256"}
            or ref["path"] != str(parent / name)
        ):
            raise HistoricalSessionError("historical_core_ref_invalid")
        raw = read(parent / name, label="historical_core_source")
        if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
            raise HistoricalSessionError("historical_core_source_sha_mismatch")
        saved[name] = raw
    proof = read_historical_fileset(path=parent / FILENAME, proof_raw=saved[FILENAME], read=read)
    raw = read(parent / RAW_FILENAME, label="historical_core_raw")
    saved[RAW_FILENAME] = raw
    close = parse_canonical_json_bytes(saved[CALENDAR_FILENAME])
    started = parse_canonical_json_bytes(saved["started.json"])
    if (
        proof["requested_trade_date"] != target
        or proof["calendar_ref"]["sha256"] != close_ref["sha256"]
        or proof["raw_calendar_ref"]["sha256"] != hashlib.sha256(raw).hexdigest()
        or close.get("raw_response_path") != str(parent / RAW_FILENAME)
        or close.get("raw_response_sha256") != proof["raw_calendar_ref"]["sha256"]
        or type(started) is not dict
        or started.get("state") != "STARTED"
        or started.get("mode") != "execute"
    ):
        raise HistoricalSessionError("historical_core_binding_invalid")
    utc_stamp(proof["observed_at"])
    utc_stamp(started.get("started_at"))
    for name, raw in saved.items():
        if read(parent / name, label="historical_core_recheck") != raw:
            raise HistoricalSessionError("historical_core_source_changed")
    after = parent.stat()
    if parent.resolve(strict=True) != parent or (before.st_dev, before.st_ino) != (
        after.st_dev,
        after.st_ino,
    ):
        raise HistoricalSessionError("historical_core_parent_changed")
    return {"historical_session": proof, "started_at": started["started_at"]}


def validate_historical_core_timing(*, observed_at, started_at, sealed_at) -> None:
    """A retry can reuse older Calendar evidence; both events must precede sealing."""
    from quant_investor.operations.daily_contract import utc_stamp

    sealed = utc_stamp(sealed_at)
    if utc_stamp(observed_at) > sealed or utc_stamp(started_at) > sealed:
        raise HistoricalSessionError("historical_core_time_invalid")


def build_historical_session(
    *, requested_trade_date: str, previous_trade_date: str, calendar_bytes: bytes, raw: bytes
) -> dict:
    calendar = parse_canonical_json_bytes(calendar_bytes)
    if type(calendar) is not dict:
        raise ValueError("HISTORICAL_SESSION_CALENDAR_INVALID")
    projection = classify_catchup_session(
        requested_trade_date=requested_trade_date,
        previous_trade_date=previous_trade_date,
        receipt=calendar,
        raw=raw,
    )
    verified = replay_close_session_authority(calendar, raw).receipt
    return {
        "schema_version": SCHEMA,
        **projection,
        "calendar_ref": {
            "path": CALENDAR_FILENAME,
            "sha256": hashlib.sha256(calendar_bytes).hexdigest(),
        },
        "raw_calendar_ref": {"path": RAW_FILENAME, "sha256": hashlib.sha256(raw).hexdigest()},
        "open_trade_dates": verified["ordered_open_dates"],
    }


def verify_historical_session(*, proof_bytes: bytes, calendar_bytes: bytes, raw: bytes) -> dict:
    proof = parse_canonical_json_bytes(proof_bytes)
    if type(proof) is not dict or proof.get("schema_version") != SCHEMA:
        raise ValueError("HISTORICAL_SESSION_SCHEMA_INVALID")
    expected = build_historical_session(
        requested_trade_date=proof.get("requested_trade_date"),
        previous_trade_date=proof.get("previous_trade_date"),
        calendar_bytes=calendar_bytes,
        raw=raw,
    )
    if canonical_json_bytes(expected) != proof_bytes:
        raise ValueError("HISTORICAL_SESSION_PROOF_MISMATCH")
    return expected
