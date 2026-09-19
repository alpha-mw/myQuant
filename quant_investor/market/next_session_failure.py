"""Typed, non-authorizing failure custody for the optional future Calendar lane."""

from datetime import datetime, timezone
import hashlib
from pathlib import PurePosixPath

from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from quant_investor.operations.daily_contract import ContractError, validate_ref, utc_stamp
from quant_investor.operations.daily_journal import DailyJournal, FALSE_AUTHORITY, _false_authority

SCHEMA = "cn-next-session-calendar-capability-failure.v1"
CODES = frozenset(
    {
        "ACQUISITION_FAILED",
        "CAPTURE_VALIDATION_FAILED",
        "HORIZON_INCOMPLETE",
        "EXCHANGE_DISAGREEMENT",
        "EOD_NOT_OPEN",
        "NO_LATER_OPEN",
        "PROVENANCE_UNAVAILABLE",
    }
)
FIELDS = frozenset(
    {
        "schema_version",
        "eod_trade_date",
        "phase",
        "failure_code",
        "execution_ref",
        "success_ref",
        "native_failure_ref",
        "failed_at",
        "authority",
    }
)


def _validate(value, day):
    if (
        type(value) is not dict
        or set(value) != FIELDS
        or value["schema_version"] != SCHEMA
        or value["eod_trade_date"] != day
        or value["phase"] not in {"ACQUISITION", "POST_CAPTURE_VALIDATION"}
        or value["failure_code"] not in CODES
        or not _false_authority(value["authority"])
    ):
        raise ContractError("NEXT_SESSION_FAILURE_CONTRACT_INVALID")
    if value["phase"] == "POST_CAPTURE_VALIDATION" and (
        value["execution_ref"] is None
        or value["success_ref"] is None
        or value["native_failure_ref"] is not None
    ):
        raise ContractError("NEXT_SESSION_FAILURE_PHASE_BINDING_INVALID")
    if utc_stamp(value["failed_at"]) > datetime.now(timezone.utc):
        raise ContractError("NEXT_SESSION_FAILURE_TIME_INVALID")
    for key, leaf in [
        ("execution_ref", "capture-execution.json"),
        ("success_ref", "capture-success.json"),
        ("native_failure_ref", "capture-failure.json"),
    ]:
        ref = value[key]
        if ref is None:
            continue
        if type(ref) is not dict or set(ref) != {"relative_path", "byte_sha256"}:
            raise ContractError("NEXT_SESSION_FAILURE_REF_INVALID")
        validate_ref({"path": ref["relative_path"], "sha256": ref["byte_sha256"]})
        path = PurePosixPath(ref["relative_path"])
        if len(path.parts) != 2 or path.name != leaf:
            raise ContractError("NEXT_SESSION_FAILURE_REF_PATH_INVALID")
    roots = {
        PurePosixPath(value[key]["relative_path"]).parent
        for key in ("execution_ref", "success_ref")
        if value[key] is not None
    }
    if len(roots) > 1:
        raise ContractError("NEXT_SESSION_FAILURE_CAPTURE_ROOT_MISMATCH")


def _read_refs(journal, value):
    for key in ("execution_ref", "success_ref", "native_failure_ref"):
        ref = value[key]
        if ref is None:
            continue
        path = str(journal.root / "calendar-future/captures" / ref["relative_path"])
        raw = journal.storage.read(path)
        if raw is None or raw.byte_sha256 != ref["byte_sha256"]:
            raise ContractError("NEXT_SESSION_FAILURE_SOURCE_SHA_MISMATCH")
        if key == "native_failure_ref":
            from .tushare_calendar_authority import (
                validate_published_trusted_provider_calendar_capture_failure_root,
            )

            native = parse_canonical_json_bytes(raw.data)
            validate_published_trusted_provider_calendar_capture_failure_root(
                capture_parent=journal.storage._io.workspace_root
                / journal.root
                / "calendar-future/captures",
                capture_failure=native,
                capture_failure_file_ref=ref,
            )
            if utc_stamp(native["payload"]["failed_at"]) > utc_stamp(value["failed_at"]):
                raise ContractError("NEXT_SESSION_FAILURE_NATIVE_CHRONOLOGY_INVALID")
            for related in (value["execution_ref"], value["success_ref"]):
                if (
                    related is not None
                    and PurePosixPath(related["relative_path"]).parent.name
                    != native["payload"]["capture_root_name"]
                ):
                    raise ContractError("NEXT_SESSION_FAILURE_CAPTURE_ROOT_MISMATCH")


def publish_next_session_failure(
    *,
    workspace: str,
    eod_trade_date: str,
    phase: str,
    failure_code: str,
    execution_ref: dict | None = None,
    success_ref: dict | None = None,
    native_failure_ref: dict | None = None,
) -> dict:
    """Record a controlled failure code, never raw exceptions or secret payloads."""
    journal = DailyJournal(workspace, eod_trade_date)
    value = {
        "schema_version": SCHEMA,
        "eod_trade_date": eod_trade_date,
        "phase": phase,
        "failure_code": failure_code,
        "execution_ref": execution_ref,
        "success_ref": success_ref,
        "native_failure_ref": native_failure_ref,
        "failed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "authority": FALSE_AUTHORITY,
    }
    _validate(value, eod_trade_date)
    _read_refs(journal, value)
    with journal.locked():
        if journal.storage.read(str(journal.root / "completion.v1.json")) is not None:
            raise ContractError("NEXT_SESSION_EOD_ALREADY_COMPLETED")
        _read_refs(journal, value)
        raw = canonical_json_bytes(value)
        digest = hashlib.sha256(raw).hexdigest()
        path = str(journal.root / "calendar-future/failures" / (digest + ".json"))
        stored = journal.storage.write(path, raw)
        ref = {"path": path, "sha256": stored.byte_sha256}
        read_next_session_failure(
            workspace=workspace, eod_trade_date=eod_trade_date, failure_ref=ref
        )
        return ref


def read_next_session_failure(*, workspace: str, eod_trade_date: str, failure_ref: dict) -> dict:
    ref = validate_ref(failure_ref)
    journal = DailyJournal(workspace, eod_trade_date)
    if ref["path"] != str(journal.root / "calendar-future/failures" / (ref["sha256"] + ".json")):
        raise ContractError("NEXT_SESSION_FAILURE_PATH_INVALID")
    stored = journal.storage.read(ref["path"])
    if stored is None or stored.byte_sha256 != ref["sha256"]:
        raise ContractError("NEXT_SESSION_FAILURE_SHA_MISMATCH")
    value = parse_canonical_json_bytes(stored.data)
    _validate(value, eod_trade_date)
    _read_refs(journal, value)
    if journal.storage.read(ref["path"]) != stored:
        raise ContractError("NEXT_SESSION_FAILURE_CHANGED_DURING_READ")
    return {"failure_ref": dict(ref), "failure": value, "consumer_admission": False}
