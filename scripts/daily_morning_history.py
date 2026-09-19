"""Historical Morning readback, with time derived only from an exact sealed receipt."""

from datetime import datetime, timezone
from pathlib import PurePosixPath
from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from quant_investor.system.storage import SecureSystemStorage
from quant_investor.operations.daily_contract import ContractError, utc_stamp, validate_ref
from quant_investor.operations.daily_journal import FALSE_AUTHORITY
from quant_investor.operations.morning_contract import validate_morning_request, request_version
from quant_investor.operations.morning_receipt import (
    validate_morning_receipt,
    receipt_path,
    receipt_version,
)
from quant_investor.operations.morning_report import validate_morning_report
from quant_investor.intelligence.morning import classify_sina_quote_timing
from scripts.daily_morning_consumer import (
    _read_morning_evidence,
    _require_evidence_at,
    _require_live_provenance,
    morning_threshold_fields,
)


def read_historical_morning_receipt(*, workspace: str, receipt_ref: dict) -> dict:
    reader = SecureSystemStorage(workspace)
    observed = {}

    def read(ref):
        checked = validate_ref(ref)
        stored = reader.read_workspace_file_bytes(checked["path"], maximum_bytes=32 * 1024 * 1024)
        if stored.byte_sha256 != checked["sha256"]:
            raise ContractError("MORNING_HISTORY_SOURCE_SHA_MISMATCH")
        observed[checked["path"]] = stored.data
        return stored.data

    recorded = validate_morning_receipt(parse_canonical_json_bytes(read(receipt_ref)))
    version = receipt_version(recorded)
    if receipt_ref["path"] != receipt_path(recorded["run_date"], version):
        raise ContractError("MORNING_RECEIPT_PATH_INVALID")
    evaluation_at = utc_stamp(recorded["validated_at"])
    observed_now = datetime.now(timezone.utc)
    if evaluation_at > observed_now:
        raise ContractError("MORNING_RECEIPT_TIME_INVALID")
    raw = read(recorded["request_ref"])
    request = validate_morning_request(parse_canonical_json_bytes(raw))
    if raw != canonical_json_bytes(request) or request["action"] != "SEAL":
        raise ContractError("MORNING_HISTORY_SEAL_REQUEST_REQUIRED")
    for key in (
        "run_date",
        "previous_completion_ref",
        "quote_capture_ref",
        "quote_raw_ref",
        "owner_policy_ref",
        "output_ref",
    ):
        if recorded[key] != request[key]:
            raise ContractError("MORNING_HISTORY_REQUEST_BINDING_INVALID")
    if request_version(request) != version:
        raise ContractError("MORNING_HISTORY_PROFILE_MISMATCH")
    report = read(request["output_ref"])
    evidence = _read_morning_evidence(workspace=workspace, request=request)
    _require_live_provenance(evidence)
    _require_evidence_at(evidence, evaluation_at)
    timing = classify_sina_quote_timing(
        evidence["quote"]["request_time"], run_date=request["run_date"]
    )
    report_context = {"quote_timing": timing}
    if version == "v3":
        if request["threshold_policy_refs"] != recorded["threshold_policy_refs"]:
            raise ContractError("MORNING_HISTORY_THRESHOLD_REFS_DIFFER")
        report_context.update(morning_threshold_fields(evidence))
        report_context.update(
            {
                k: request[k]
                for k in (
                    "run_date",
                    "previous_completion_ref",
                    "quote_capture_ref",
                    "quote_raw_ref",
                )
            }
        )
        report_context.update(
            previous_trade_date=recorded["previous_trade_date"],
            decision=evidence["native"]["decision"],
            admission=recorded["admission"],
        )
    validate_morning_report(report, report_context)
    rebuilt = {
        "schema_version": "morning-strategy-run." + version,
        "run_date": request["run_date"],
        "previous_trade_date": PurePosixPath(
            request["previous_completion_ref"]["path"]
        ).parent.name,
        "request_ref": recorded["request_ref"],
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
        "status": "COMPLETE",
        "admission": "LIVE_RESEARCH_CONSUMER",
        "synthetic": False,
        "prospective_admission_state": "NOT_CLAIMED",
        "expected_symbols": evidence["symbols"],
        "quote_timing": timing,
        "authority": dict(FALSE_AUTHORITY),
    }
    if version == "v3":
        rebuilt.update(
            threshold_policy_refs=request["threshold_policy_refs"],
            threshold_review_sha256=report_context["threshold_review"]["content_sha256"],
            review_summary_state=report_context["threshold_review"]["summary_state"],
        )
    if rebuilt != {key: value for key, value in recorded.items() if key != "validated_at"}:
        raise ContractError("MORNING_HISTORY_RECEIPT_SEMANTICS_INVALID")
    evidence["recheck"]()
    for path, content in observed.items():
        if reader.read_workspace_file_bytes(path, maximum_bytes=32 * 1024 * 1024).data != content:
            raise ContractError("MORNING_HISTORY_SOURCE_CHANGED")
    if datetime.now(timezone.utc) < observed_now:
        raise ContractError("MORNING_HISTORY_CLOCK_REGRESSED")
    return {
        "schema_version": "morning-strategy-seal-result." + version,
        "command_status": "RECEIPT_VERIFIED",
        "receipt_ref": dict(receipt_ref),
        "receipt": recorded,
    }
