"""Public daily-close boundary over fixed verified installed orchestration."""

from pathlib import PurePosixPath
import re
from quant_investor.cli.input import read_exact_request
from quant_investor.cli.output import CommandError
from quant_investor.operations.daily_contract import ContractError, EOD_NODE_IDS
from quant_investor.operations.production_request import validate_production_request
from quant_investor.operations.production_result import validate_production_result
from quant_investor.operations.native_bridge import verified_native_context
from quant_investor.operations.catchup import plan_catchup, catchup_result_dates
from quant_investor.operations.automatic_catchup_contract import (
    REQUEST_SCHEMA as AUTO_REQUEST_SCHEMA,
    REQUEST_SCHEMA_V2 as AUTO_REQUEST_SCHEMA_V2,
    validate_automatic_request,
    validate_automatic_result,
    AutomaticCatchupError,
)


def run_daily_close(
    *,
    workspace: str,
    request_path: str,
    expected_request_sha256: str,
    release_repository_root: str | None,
    release_install_input_path: str | None,
    expected_release_install_input_sha256: str | None,
    no_producers: bool = False,
) -> dict:
    if not all(
        type(value) is str and value
        for value in (
            release_repository_root,
            release_install_input_path,
            expected_release_install_input_sha256,
        )
    ):
        raise CommandError("DAILY_PRODUCTION_RELEASE_ARGUMENTS_REQUIRED")
    install_ref = {
        "path": release_install_input_path,
        "sha256": expected_release_install_input_sha256,
    }
    _, document = read_exact_request(workspace, request_path, expected_request_sha256)
    automatic = (
        type(document) is dict
        and type(document.get("schema_version")) is str
        and document["schema_version"]
        in {
            AUTO_REQUEST_SCHEMA,
            AUTO_REQUEST_SCHEMA_V2,
        }
    )
    try:
        request = (validate_automatic_request if automatic else validate_production_request)(
            document, release_install_ref=install_ref
        )
    except (ValueError, TypeError) as exc:
        raise CommandError("DAILY_PRODUCTION_REQUEST_INVALID") from exc
    if type(no_producers) is not bool or (
        no_producers and (not automatic or request["action"] != "CATCH_UP")
    ):
        raise CommandError("AUTO_NO_PRODUCERS_FLAG_INVALID")
    raw, _ = read_exact_request(
        workspace, release_install_input_path or "", expected_release_install_input_sha256 or ""
    )
    try:
        with verified_native_context(
            release_input_raw=raw,
            expected_sha256=expected_release_install_input_sha256 or "",
            repository_root=release_repository_root or "",
        ) as operations:
            try:
                if automatic:
                    return _run_automatic_result(
                        workspace,
                        {"path": request_path, "sha256": expected_request_sha256},
                        request,
                        install_ref,
                        operations,
                        no_producers=no_producers,
                    )
                dates = _legacy_result_dates(workspace, request)
                result = operations["daily_close"](
                    workspace=workspace,
                    request_ref={"path": request_path, "sha256": expected_request_sha256},
                    release_install_ref=install_ref,
                )
            except (ValueError, OSError) as exc:
                _evidence_error(exc, automatic=automatic)
            return _validate_daily_result(result, workspace, request, dates, operations)
    except ContractError as exc:
        raise CommandError("DAILY_PRODUCTION_RUNTIME_REJECTED") from exc


def _legacy_result_dates(workspace, request):
    day = request["target_trade_date"]
    if request["action"] != "CATCH_UP":
        return [day]
    previous = PurePosixPath(request["previous_completion_ref"]["path"]).parent.name
    planned = plan_catchup(
        workspace=workspace,
        calendar_ref=request["calendar_ref"],
        raw_calendar_ref=request["raw_calendar_ref"],
        previous_trade_date=previous,
        target_trade_date=day,
        day_input_refs=request["day_input_refs"],
    )
    return catchup_result_dates(planned, previous_trade_date=previous, target_trade_date=day)


def _evidence_error(exc, *, automatic):
    # The installed class can be a distinct Python object from the host class.
    # Validate the small wire-equivalent error contract rather than its identity.
    code = getattr(exc, "code", str(exc))
    if (
        automatic
        and isinstance(exc, ValueError)
        and type(code) is str
        and re.fullmatch(r"AUTO_[A-Z0-9_]+", code)
    ):
        fields = getattr(exc, "fields", {})
        if type(fields) is not dict or set(fields) - {
            "pending_request_ref",
            "missing_input_dates",
            "completed_eod_refs",
        }:
            raise RuntimeError("native automatic error fields invalid") from exc
        try:
            checked = AutomaticCatchupError(code, **fields)
        except (ValueError, TypeError) as invalid:
            raise RuntimeError("native automatic error fields invalid") from invalid
        raise CommandError(code, fields=checked.fields) from exc
    raise CommandError("DAILY_PRODUCTION_EVIDENCE_REJECTED") from exc


def _validate_daily_result(result, workspace, request, dates, operations):
    try:
        if request["action"] == "EXECUTE" and result.get("business_state") == "NON_TRADING_DAY":
            dates = []
        validate_production_result(
            result,
            action=request["action"],
            target_trade_date=request["target_trade_date"],
            expected_dates=dates,
        )
        for row in result["days"]:
            if row["completion_ref"] is not None:
                replay = operations["completion_replay"](
                    workspace=workspace,
                    trade_date=row["trade_date"],
                    completion_ref=row["completion_ref"],
                )
                if (
                    replay.get("native_replay_validated") is not True
                    or replay.get("completion_ref") != row["completion_ref"]
                    or replay.get("trade_date") != row["trade_date"]
                    or replay.get("validated_nodes") != sorted(EOD_NODE_IDS)
                    or replay.get("synthetic") is not False
                ):
                    raise ValueError("native completion result is invalid")
    except Exception as exc:
        raise RuntimeError("native daily-close result validation failed") from exc
    return result


def _run_automatic_result(
    workspace, request_ref, request, install_ref, operations, *, no_producers=False
):
    result = operations["daily_close"](
        workspace=workspace,
        request_ref=request_ref,
        release_install_ref=install_ref,
        **({"no_producers": True} if no_producers else {}),
    )
    return validate_native_automatic_result(
        result, workspace, request_ref, request, install_ref, operations
    )


def validate_native_automatic_result(
    result, workspace, request_ref, request, install_ref, operations
):
    """Shared installed readback for public dispatch and completed launcher inspection."""
    try:
        if request["action"] == "PLAN":
            # The installed read-only operation can independently derive a PLAN candidate.
            resolution = operations["automatic_resolution"](
                workspace=workspace,
                request_ref=request_ref,
                resolution_ref=None,
                release_install_ref=install_ref,
            )
        else:
            resolution = operations["automatic_resolution"](
                workspace=workspace,
                resolution_ref=result["resolution_ref"],
                release_install_ref=install_ref,
            )
        validate_automatic_result(
            result, request_ref=request_ref, request=request, resolution=resolution
        )
        if request["action"] == "CATCH_UP":
            for row in result["result"]["days"]:
                if row["completion_ref"] is None:
                    continue
                replay = operations["completion_replay"](
                    workspace=workspace,
                    trade_date=row["trade_date"],
                    completion_ref=row["completion_ref"],
                )
                if (
                    replay.get("native_replay_validated") is not True
                    or replay.get("completion_ref") != row["completion_ref"]
                    or replay.get("trade_date") != row["trade_date"]
                    or replay.get("validated_nodes") != sorted(EOD_NODE_IDS)
                    or replay.get("synthetic") is not False
                ):
                    raise ValueError("invalid automatic completion replay")
    except Exception as exc:
        raise RuntimeError("native automatic daily-close result validation failed") from exc
    return result
