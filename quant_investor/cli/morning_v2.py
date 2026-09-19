"""Installed-only Morning v2 dispatch over the fixed native consumer."""

from pathlib import PurePosixPath
from datetime import datetime, timezone

from quant_investor.cli.input import read_exact_request
from quant_investor.cli.output import CommandError
from quant_investor.operations.daily_contract import ContractError, utc_stamp, validate_ref
from quant_investor.operations.daily_journal import _false_authority
from quant_investor.operations.morning_contract import validate_morning_request, request_version
from quant_investor.operations.native_bridge import verified_native_context
from quant_investor.system.storage import SecureSystemStorage

RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "command_status",
        "admission",
        "run_date",
        "previous_trade_date",
        "previous_completion_ref",
        "expected_symbols",
        "quote_capture_ref",
        "quote_raw_ref",
        "owner_policy_ref",
        "quote_rows",
        "quote_timing",
        "validated_at",
        "synthetic",
        "prospective_admission_state",
        "decision",
        "authority",
    }
)


V3_RESULT_FIELDS = {"threshold_policy_refs", "threshold_review", "report_markdown"}


def _validate_result(value: dict, request: dict, workspace: str) -> dict:
    from quant_investor.intelligence.morning import (
        validate_sina_quote_capture,
        classify_sina_quote_timing,
    )

    version = request_version(request)
    if (
        type(value) is not dict
        or set(value) != RESULT_FIELDS | (V3_RESULT_FIELDS if version == "v3" else set())
        or value["schema_version"] != "morning-strategy-replay." + version
        or request["action"] not in {"REPLAY", "PREFLIGHT"}
        or value["command_status"]
        != ("REPLAY_VERIFIED" if request["action"] == "REPLAY" else "PREFLIGHT_COMPLETE")
        or value["admission"]
        != ("RESEARCH_ONLY" if request["action"] == "REPLAY" else "LIVE_RESEARCH_CONSUMER")
        or (request["action"] == "PREFLIGHT" and value["synthetic"] is not False)
        or value["prospective_admission_state"] != "NOT_CLAIMED"
        or type(value["synthetic"]) is not bool
        or not _false_authority(value["authority"])
        or type(value["decision"]) is not dict
    ):
        raise ValueError("invalid native Morning result")
    for key in (
        "run_date",
        "previous_completion_ref",
        "quote_capture_ref",
        "quote_raw_ref",
        "owner_policy_ref",
    ):
        if value[key] != request[key]:
            raise ValueError("native Morning result request mismatch")
    if (
        value["previous_trade_date"]
        != PurePosixPath(request["previous_completion_ref"]["path"]).parent.name
    ):
        raise ValueError("native Morning result date mismatch")
    if utc_stamp(value["validated_at"]) > datetime.now(timezone.utc):
        raise ValueError("native Morning result time invalid")
    _, capture = read_exact_request(
        workspace, request["quote_capture_ref"]["path"], request["quote_capture_ref"]["sha256"]
    )
    raw = SecureSystemStorage(workspace).read_workspace_file_bytes(
        request["quote_raw_ref"]["path"], maximum_bytes=16 * 1024 * 1024
    )
    if raw.byte_sha256 != request["quote_raw_ref"]["sha256"]:
        raise ValueError("native Morning result quote changed")
    quote = validate_sina_quote_capture(capture, raw=raw.data, run_date=request["run_date"])
    if (
        value["quote_rows"] != quote["quote_rows"]
        or value["expected_symbols"] != [row["symbol"] for row in quote["quote_rows"]]
        or value["quote_timing"]
        != classify_sina_quote_timing(quote["request_time"], run_date=request["run_date"])
    ):
        raise ValueError("native Morning result quote mismatch")
    if version == "v3":
        _validate_threshold_result(value, request, workspace, quote)
    return value


def run_morning_v2(
    *,
    workspace: str,
    request: dict,
    request_ref: dict | None = None,
    release_repository_root: str,
    release_install_input_path: str,
    expected_release_install_input_sha256: str,
) -> dict:
    try:
        values = validate_morning_request(request)
    except (ValueError, TypeError) as exc:
        raise CommandError("MORNING_V2_REQUEST_INVALID") from exc
    checked_request = (
        _read_seal_request(workspace, values, request_ref) if values["action"] == "SEAL" else None
    )
    raw, _ = read_exact_request(
        workspace, release_install_input_path, expected_release_install_input_sha256
    )
    try:
        with verified_native_context(
            release_input_raw=raw,
            expected_sha256=expected_release_install_input_sha256,
            repository_root=release_repository_root,
        ) as operations:
            result = _invoke_morning(operations, workspace, values, checked_request)
            # Result failures are implementation errors (exit 3), outside expected
            # business error mapping. Native context cleanup still always runs.
            try:
                if values["action"] == "SEAL":
                    return _validate_seal_result(
                        result, values, checked_request, workspace, operations
                    )
                return _validate_result(result, values, workspace)
            except Exception as exc:
                raise RuntimeError("native Morning result validation failed") from exc
    except ContractError as exc:
        raise CommandError("MORNING_V2_RUNTIME_REJECTED") from exc


def _validate_seal_result(value, request, request_ref, workspace, operations):
    from quant_investor.operations.morning_receipt import validate_morning_receipt, receipt_path

    version = request_version(request)
    fields = {"schema_version", "command_status", "receipt_ref", "receipt"}
    if (
        type(value) is not dict
        or set(value) != fields
        or value["schema_version"] != "morning-strategy-seal-result." + version
        or value["command_status"] not in {"PUBLISHED", "NO_ACTION"}
    ):
        raise ValueError("invalid native Morning seal result")
    reference = validate_ref(value["receipt_ref"])
    receipt = validate_morning_receipt(value["receipt"])
    if (
        reference["path"] != receipt_path(request["run_date"], version)
        or receipt["request_ref"] != request_ref
    ):
        raise ValueError("native Morning seal request mismatch")
    for key in (
        "run_date",
        "previous_completion_ref",
        "quote_capture_ref",
        "quote_raw_ref",
        "owner_policy_ref",
        "output_ref",
    ):
        if receipt[key] != request[key]:
            raise ValueError("native Morning seal binding mismatch")
    if version == "v3" and receipt["threshold_policy_refs"] != request["threshold_policy_refs"]:
        raise ValueError("native Morning threshold receipt binding mismatch")
    checked = operations["morning_receipt"](workspace=workspace, receipt_ref=reference)
    if (
        type(checked) is not dict
        or set(checked) != fields
        or checked["schema_version"] != "morning-strategy-seal-result." + version
        or checked["command_status"] != "RECEIPT_VERIFIED"
        or checked["receipt_ref"] != reference
        or checked["receipt"] != receipt
    ):
        raise ValueError("native Morning receipt replay failed")
    return value


def _read_seal_request(workspace, values, request_ref):
    try:
        checked_request = validate_ref(request_ref)
    except (ValueError, TypeError) as exc:
        raise CommandError("MORNING_V2_REQUEST_REF_REQUIRED") from exc
    _, original_request = read_exact_request(
        workspace, checked_request["path"], checked_request["sha256"]
    )
    if original_request != values:
        raise CommandError("MORNING_V2_REQUEST_REF_MISMATCH")
    return checked_request


def _invoke_morning(operations, workspace, values, checked_request):
    try:
        if values["action"] == "SEAL":
            result = operations["morning_seal"](workspace=workspace, request_ref=checked_request)
        else:
            result = operations["morning"](workspace=workspace, request=values)
    except (ValueError, OSError) as exc:
        code = str(exc)
        if request_version(values) == "v3" and code == "MORNING_UPSTREAM_DAG_INCOMPLETE":
            from quant_investor.operations.daily_contract import EOD_NODE_IDS

            nodes = getattr(exc, "failed_node_ids", [])
            if type(nodes) is not list or any(node not in EOD_NODE_IDS for node in nodes):
                nodes = []
            raise CommandError(
                code,
                fields={
                    "previous_trade_date": PurePosixPath(
                        values["previous_completion_ref"]["path"]
                    ).parent.name,
                    "failed_node_ids": sorted(set(nodes)),
                    "failure_scope": "NATIVE_EOD_ADMISSION",
                },
            ) from exc
        allowed = {
            "CALENDAR_NEXT_SESSION_UNAVAILABLE",
            "MORNING_V2_LIVE_ADMISSION_NOT_READY",
            "MORNING_LIVE_PROVENANCE_REQUIRED",
            "MORNING_LIVE_DATE_MISMATCH",
        }
        raise CommandError(code if code in allowed else "MORNING_V2_EVIDENCE_REJECTED") from exc
    return result


def _validate_threshold_result(value, request, workspace, quote):
    from quant_investor.intelligence.morning_threshold_review import validate_threshold_review
    from quant_investor.operations.morning_report import render_morning_report
    from quant_investor.strategy_records.event_receipts import read_event_source

    review = validate_threshold_review(value["threshold_review"])
    if value["threshold_policy_refs"] != request["threshold_policy_refs"]:
        raise ValueError("native Morning owner policy refs differ")
    for key in (
        "run_date",
        "previous_trade_date",
        "previous_completion_ref",
        "quote_capture_ref",
        "quote_raw_ref",
        "threshold_policy_refs",
        "synthetic",
    ):
        if review[key] != value[key]:
            raise ValueError("native Morning threshold identity differs")
    if review["quote_requested_at"] != quote["request_time"]:
        raise ValueError("native Morning threshold quote clock differs")
    if (
        sorted([row["symbol"] for row in review["rows"]] + review["quote_only_symbols"])
        != value["expected_symbols"]
    ):
        raise ValueError("native Morning threshold symbol set differs")
    reference = request["previous_completion_ref"]
    _, recorded = read_exact_request(workspace, reference["path"], reference["sha256"])
    terminal_ref = recorded["node_terminal_refs"]["decision"]
    _, terminal = read_exact_request(workspace, terminal_ref["path"], terminal_ref["sha256"])
    if (
        review["decision_result_ref"] != terminal["output_refs"]["result"]
        or review["eod_validated_at"] != recorded["native_validation_completed_at"]
    ):
        raise ValueError("native Morning threshold EOD binding differs")
    from quant_investor.contracts import parse_canonical_json_bytes

    if (
        parse_canonical_json_bytes(read_event_source(workspace, review["decision_result_ref"]))
        != value["decision"]
    ):
        raise ValueError("native Morning Decision content differs")
    for ref in review["source_refs"]:
        read_event_source(workspace, ref)
    if type(value["report_markdown"]) is not str or value["report_markdown"].encode(
        "utf-8"
    ) != render_morning_report(value):
        raise ValueError("native Morning report content differs")
