"""Native v2 cutover recommendation; no scheduler or maintenance operation."""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from quant_investor.contracts import parse_canonical_json_bytes
from quant_investor.system.storage import SecureSystemStorage
from quant_investor.system.errors import SystemImmutableConflict
from quant_investor.operations.daily_contract import ContractError, utc_stamp, validate_ref
from quant_investor.operations.daily_journal import FALSE_AUTHORITY, _false_authority
from quant_investor.operations.morning_receipt import validate_morning_receipt, receipt_version
from quant_investor.operations.morning_cutover_contract import (
    validate_cutover_request,
    validate_cutover_receipt,
    CutoverStorage,
    SCHEMA,
    cutover_path,
)
from quant_investor.intelligence.morning import _schedule_transition
from scripts.daily_ledger_replay import select_eligible_daily_evidence
from scripts.daily_morning_seal import read_morning_consumer_receipt


def recommend_morning_cutover(*, workspace: str, request_ref: dict, release_install_ref: dict):
    reader = SecureSystemStorage(workspace)
    observed = {}

    def read(ref):
        checked = validate_ref(ref)
        raw = reader.read_workspace_file_bytes(checked["path"], maximum_bytes=32 * 1024 * 1024)
        if raw.byte_sha256 != checked["sha256"]:
            raise ContractError("MORNING_CUTOVER_SOURCE_SHA_MISMATCH")
        prior = observed.get(checked["path"])
        if prior is not None and prior != raw.data:
            raise ContractError("MORNING_CUTOVER_SOURCE_CHANGED")
        observed[checked["path"]] = raw.data
        return parse_canonical_json_bytes(raw.data)

    request = validate_cutover_request(read(request_ref))
    read(release_install_ref)  # Outer bridge has verified the installation, not scheduler state.
    day = request["target_date"]
    now = datetime.now(timezone.utc)
    if now.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d") != day:
        raise ContractError("MORNING_CUTOVER_LIVE_DATE_REQUIRED")
    selection = select_eligible_daily_evidence(
        workspace=workspace, trade_date=day, completion_ref=request["daily_completion_ref"]
    )
    fields = {
        "schema_version",
        "market",
        "trade_date",
        "completion_ref",
        "ledger_ref",
        "eligibility_scope",
        "factor_admission",
        "authority",
    }
    if (
        type(selection) is not dict
        or set(selection) != fields
        or selection["schema_version"] != "cn-daily-eligible-evidence-selection.v1"
        or selection["market"] != "CN"
        or selection["trade_date"] != day
        or selection["completion_ref"] != request["daily_completion_ref"]
        or selection["eligibility_scope"] != "LOCAL_COORDINATOR_AVAILABILITY"
        or selection["factor_admission"] is not False
        or not _false_authority(selection["authority"])
    ):
        raise ContractError("MORNING_CUTOVER_NATIVE_SELECTION_INVALID")
    completion = read(request["daily_completion_ref"])
    ledger = read(selection["ledger_ref"])
    if (
        completion["prospective_ledger_ref"] != selection["ledger_ref"]
        or ledger["schema_version"] != "cn-daily-evidence-ledger.v1"
        or ledger["trade_date"] != day
        or ledger["classification"] != "CONTEMPORANEOUS"
        or ledger["prospective"] is not True
        or ledger["synthetic"] is not False
        or ledger["recomputed"] is not False
        or not _false_authority(ledger["authority"])
    ):
        raise ContractError("MORNING_CUTOVER_LEDGER_INELIGIBLE")
    receipts = []
    for ref in request["morning_receipts"]:
        result = read_morning_consumer_receipt(workspace=workspace, receipt_ref=ref)
        if (
            type(result) is not dict
            or set(result) != {"schema_version", "command_status", "receipt_ref", "receipt"}
            or result["schema_version"]
            != "morning-strategy-seal-result." + receipt_version(result["receipt"])
            or result["command_status"] != "RECEIPT_VERIFIED"
            or result["receipt_ref"] != ref
        ):
            raise ContractError("MORNING_CUTOVER_RECEIPT_REPLAY_INVALID")
        receipt = validate_morning_receipt(read(ref))
        if receipt != result["receipt"]:
            raise ContractError("MORNING_CUTOVER_RECEIPT_CHANGED")
        receipts.append(receipt)
    has_today = bool(receipts and receipts[-1]["run_date"] == day)
    count = int(has_today)
    if (
        has_today
        and len(receipts) == 2
        and receipts[0]["run_date"] == receipts[1]["previous_trade_date"]
    ):
        count = 2
    next_state, action = _schedule_transition(
        True, request["current_schedule_state"], count, has_today
    )

    def recheck():
        for path, raw in observed.items():
            if reader.read_workspace_file_bytes(path, maximum_bytes=32 * 1024 * 1024).data != raw:
                raise ContractError("MORNING_CUTOVER_SOURCE_CHANGED")

    recheck()
    stamp = datetime.now(timezone.utc)
    lower = max(
        [
            utc_stamp(completion["native_validation_completed_at"]),
            *[utc_stamp(r["validated_at"]) for r in receipts],
        ]
    )
    if (
        stamp < now
        or stamp < lower
        or stamp.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d") != day
    ):
        raise ContractError("MORNING_CUTOVER_TIME_INVALID")
    value = {
        "schema_version": SCHEMA,
        "target_date": day,
        "request_ref": dict(request_ref),
        "daily_completion_ref": request["daily_completion_ref"],
        "morning_receipts": request["morning_receipts"],
        "release_install_ref": dict(release_install_ref),
        "current_schedule_state": request["current_schedule_state"],
        "current_schedule_state_basis": "OWNER_DECLARATION",
        "next_schedule_state": next_state,
        "schedule_action": action,
        "consecutive_morning_success_count": count,
        "validated_at": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "COMPLETE",
        "admission": "SCHEDULE_RECOMMENDATION_ONLY",
        "application_performed": False,
        "authority": dict(FALSE_AUTHORITY),
    }
    validate_cutover_receipt(value)
    storage = CutoverStorage(workspace)

    def returned(stored, status):
        old = validate_cutover_receipt(parse_canonical_json_bytes(stored.data))
        if {k: v for k, v in old.items() if k != "validated_at"} != {
            k: v for k, v in value.items() if k != "validated_at"
        } or not lower <= utc_stamp(old["validated_at"]) <= stamp:
            raise ContractError("MORNING_CUTOVER_IMMUTABLE_CONFLICT")
        recheck()
        if storage.read(day) != stored:
            raise ContractError("MORNING_CUTOVER_RECEIPT_CHANGED")
        return {
            "schema_version": "morning-strategy-cutover-result.v2",
            "command_status": status,
            "receipt_ref": {"path": cutover_path(day), "sha256": stored.byte_sha256},
            "receipt": old,
        }

    old = storage.read(day)
    if old is not None:
        return returned(old, "NO_ACTION")
    try:
        stored = storage.write(value)
    except SystemImmutableConflict:
        old = storage.read(day)
        if old is None:
            raise
        return returned(old, "NO_ACTION")
    return returned(stored, "PUBLISHED")
