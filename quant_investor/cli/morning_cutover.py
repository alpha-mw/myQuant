"""Installed v2 cutover routing; recommendation never applies scheduler state."""

from quant_investor.cli.input import read_exact_request
from quant_investor.cli.output import CommandError
from quant_investor.operations.daily_contract import ContractError, validate_ref
from quant_investor.operations.morning_cutover_contract import (
    validate_cutover_request,
    validate_cutover_receipt,
    cutover_path,
)
from quant_investor.operations.native_bridge import verified_native_context


def _validate_result(result, request, request_ref, install_ref, workspace):
    if (
        type(result) is not dict
        or set(result) != {"schema_version", "command_status", "receipt_ref", "receipt"}
        or result["schema_version"] != "morning-strategy-cutover-result.v2"
        or result["command_status"] not in {"PUBLISHED", "NO_ACTION"}
    ):
        raise ValueError("invalid native cutover result")
    receipt = validate_cutover_receipt(result["receipt"])
    if receipt["request_ref"] != request_ref or receipt["release_install_ref"] != install_ref:
        raise ValueError("cutover request/runtime binding invalid")
    for key in (
        "target_date",
        "daily_completion_ref",
        "morning_receipts",
        "current_schedule_state",
    ):
        if receipt[key] != request[key]:
            raise ValueError("cutover input binding invalid")
    ref = validate_ref(result["receipt_ref"])
    if ref["path"] != cutover_path(request["target_date"]):
        raise ValueError("cutover receipt path invalid")
    _, stored = read_exact_request(workspace, ref["path"], ref["sha256"])
    if stored != receipt:
        raise ValueError("cutover receipt readback invalid")
    return result


def run_morning_cutover_v2(
    *,
    workspace,
    request_path,
    expected_request_sha256,
    release_repository_root,
    release_install_input_path,
    expected_release_install_input_sha256,
):
    if not all(
        type(v) is str and v
        for v in (
            release_repository_root,
            release_install_input_path,
            expected_release_install_input_sha256,
        )
    ):
        raise CommandError("MORNING_CUTOVER_RELEASE_ARGUMENTS_REQUIRED")
    _, document = read_exact_request(workspace, request_path, expected_request_sha256)
    try:
        request = validate_cutover_request(document)
    except (ValueError, TypeError) as exc:
        raise CommandError("MORNING_CUTOVER_REQUEST_INVALID") from exc
    request_ref = {"path": request_path, "sha256": expected_request_sha256}
    install_ref = {
        "path": release_install_input_path,
        "sha256": expected_release_install_input_sha256,
    }
    raw, _ = read_exact_request(
        workspace, release_install_input_path, expected_release_install_input_sha256
    )
    try:
        with verified_native_context(
            release_input_raw=raw,
            expected_sha256=expected_release_install_input_sha256,
            repository_root=release_repository_root,
        ) as operations:
            try:
                result = operations["morning_cutover"](
                    workspace=workspace, request_ref=request_ref, release_install_ref=install_ref
                )
            except (ValueError, OSError) as exc:
                raise CommandError("MORNING_CUTOVER_EVIDENCE_REJECTED") from exc
            try:
                return _validate_result(result, request, request_ref, install_ref, workspace)
            except Exception as exc:
                raise RuntimeError("native cutover result validation failed") from exc
    except ContractError as exc:
        raise CommandError("MORNING_CUTOVER_RUNTIME_REJECTED") from exc
