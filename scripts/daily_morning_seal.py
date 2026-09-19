"""Seal only Morning's own receipt after exact request/report and live native replay."""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from quant_investor.system.storage import SecureSystemStorage
from quant_investor.system.errors import SystemImmutableConflict
from quant_investor.operations.daily_contract import ContractError, utc_stamp, validate_ref
from quant_investor.operations.daily_journal import FALSE_AUTHORITY
from quant_investor.operations.morning_contract import validate_morning_request, request_version
from quant_investor.operations.morning_receipt import (
    MorningReceiptStorage,
    receipt_version,
    receipt_path,
    validate_morning_receipt,
)
from scripts.daily_morning_consumer import prepare_morning_consumer
from quant_investor.operations.morning_report import validate_morning_report


def _prepare_seal_context(*, workspace: str, request_ref: dict):
    reader = SecureSystemStorage(workspace)
    observed = {}

    def read(ref):
        checked = validate_ref(ref)
        stored = reader.read_workspace_file_bytes(checked["path"], maximum_bytes=32 * 1024 * 1024)
        if stored.byte_sha256 != checked["sha256"]:
            raise ContractError("MORNING_SEAL_SOURCE_SHA_MISMATCH")
        previous = observed.get(checked["path"])
        if previous is not None and previous != stored.data:
            raise ContractError("MORNING_SEAL_SOURCE_CHANGED")
        observed[checked["path"]] = stored.data
        return stored.data

    raw_request = read(request_ref)
    request = validate_morning_request(parse_canonical_json_bytes(raw_request))
    if raw_request != canonical_json_bytes(request) or request["action"] != "SEAL":
        raise ContractError("MORNING_SEAL_REQUEST_REQUIRED")
    version = request_version(request)
    report = read(request["output_ref"])
    completion = parse_canonical_json_bytes(read(request["previous_completion_ref"]))
    capture = parse_canonical_json_bytes(read(request["quote_capture_ref"]))
    read(request["quote_raw_ref"])
    read(request["owner_policy_ref"])
    native_policy_bytes = {}
    if version == "v3":
        from quant_investor.strategy_records.event_receipts import read_event_source

        for ref in request["threshold_policy_refs"].values():
            native_policy_bytes[(ref["path"], ref["sha256"])] = read_event_source(workspace, ref)
    prepared = prepare_morning_consumer(
        workspace=workspace, request={**request, "action": "PREFLIGHT", "output_ref": None}
    )
    if (
        prepared.get("command_status") != "PREFLIGHT_COMPLETE"
        or prepared.get("admission") != "LIVE_RESEARCH_CONSUMER"
        or prepared.get("synthetic") is not False
        or prepared.get("prospective_admission_state") != "NOT_CLAIMED"
    ):
        raise ContractError("MORNING_SEAL_LIVE_PREFLIGHT_REQUIRED")
    for key in (
        "run_date",
        "previous_completion_ref",
        "quote_capture_ref",
        "quote_raw_ref",
        "owner_policy_ref",
    ):
        if prepared[key] != request[key]:
            raise ContractError("MORNING_SEAL_PREFLIGHT_BINDING_INVALID")
    if (
        version == "v3"
        and prepared.get("threshold_policy_refs") != request["threshold_policy_refs"]
    ):
        raise ContractError("MORNING_SEAL_THRESHOLD_BINDING_INVALID")
    validate_morning_report(report, prepared)
    lower = max(
        utc_stamp(completion["native_validation_completed_at"]), utc_stamp(capture["response_time"])
    )

    def recheck():
        for (path, sha), raw in native_policy_bytes.items():
            if read_event_source(workspace, {"path": path, "sha256": sha}) != raw:
                raise ContractError("MORNING_SEAL_POLICY_SOURCE_CHANGED")
        for path, raw in observed.items():
            if reader.read_workspace_file_bytes(path, maximum_bytes=32 * 1024 * 1024).data != raw:
                raise ContractError("MORNING_SEAL_SOURCE_CHANGED")

    recheck()
    now = datetime.now(timezone.utc)
    if (
        now < lower
        or now < utc_stamp(prepared["validated_at"])
        or now.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d") != request["run_date"]
    ):
        raise ContractError("MORNING_SEAL_CLOCK_INVALID")
    value = {
        "schema_version": "morning-strategy-run." + version,
        **{
            key: prepared[key]
            for key in ("run_date", "previous_trade_date", "expected_symbols", "quote_timing")
        },
        "request_ref": dict(request_ref),
        **{
            key: request[key]
            for key in (
                "previous_completion_ref",
                "quote_capture_ref",
                "quote_raw_ref",
                "owner_policy_ref",
                "output_ref",
            )
        },
        "validated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "COMPLETE",
        "admission": "LIVE_RESEARCH_CONSUMER",
        "synthetic": False,
        "prospective_admission_state": "NOT_CLAIMED",
        "authority": dict(FALSE_AUTHORITY),
    }
    if version == "v3":
        value.update(
            threshold_policy_refs=request["threshold_policy_refs"],
            threshold_review_sha256=prepared["threshold_review"]["content_sha256"],
            review_summary_state=prepared["threshold_review"]["summary_state"],
        )
    validate_morning_receipt(value)
    storage = MorningReceiptStorage(workspace)

    def result(stored, status):
        retained = validate_morning_receipt(parse_canonical_json_bytes(stored.data))
        if {k: v for k, v in retained.items() if k != "validated_at"} != {
            k: v for k, v in value.items() if k != "validated_at"
        } or not lower <= utc_stamp(retained["validated_at"]) <= now:
            raise ContractError("MORNING_SEAL_RECEIPT_CONFLICT")
        recheck()
        if storage.read(request["run_date"], version) != stored:
            raise ContractError("MORNING_SEAL_RECEIPT_CHANGED")
        return {
            "schema_version": "morning-strategy-seal-result." + version,
            "command_status": status,
            "receipt_ref": {
                "path": receipt_path(request["run_date"], version),
                "sha256": stored.byte_sha256,
            },
            "receipt": retained,
        }

    return storage, value, result


def seal_morning_consumer(*, workspace: str, request_ref: dict) -> dict:
    storage, value, result = _prepare_seal_context(workspace=workspace, request_ref=request_ref)
    existing = storage.read(value["run_date"], receipt_version(value))
    if existing is not None:
        return result(existing, "NO_ACTION")
    try:
        stored = storage.write(value)
    except SystemImmutableConflict:
        # A concurrent valid publisher may win; compare all semantics and retain its time.
        existing = storage.read(value["run_date"], receipt_version(value))
        if existing is None:
            raise
        return result(existing, "NO_ACTION")
    return result(stored, "PUBLISHED")


def read_morning_consumer_receipt(*, workspace: str, receipt_ref: dict) -> dict:
    """Historical receipt-only route; never re-enters live admission or publication."""
    from scripts.daily_morning_history import read_historical_morning_receipt

    return read_historical_morning_receipt(workspace=workspace, receipt_ref=receipt_ref)
