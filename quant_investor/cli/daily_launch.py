"""Fixed installed launcher inspection/reader; internal10/11 never become public run exits."""

from contextlib import contextmanager
from pathlib import PurePosixPath
import re
import sys

from quant_investor.contracts import canonical_json_bytes
from quant_investor.cli.input import read_exact_request
from quant_investor.cli.output import CommandError, MachineArgumentParser, command_boundary
from quant_investor.cli.daily_production import (
    validate_native_automatic_result,
    _validate_daily_result,
    _evidence_error,
)
from quant_investor.operations.automatic_catchup_contract import validate_automatic_request
from quant_investor.operations.daily_launch_contract import (
    MODES,
    BOOTSTRAP_SCHEMA,
    REGISTERED_SCHEMA,
    validate_launch_inspection,
)
from quant_investor.operations.bootstrap_launch_contract import BootstrapLaunchInputs
from quant_investor.operations.automatic_catchup_contract import automatic_result_exit_code
from quant_investor.operations.production_result import production_result_exit_code
from quant_investor.operations.native_bridge import verified_native_context


@contextmanager
def _installed(args):
    request_ref = {"path": args.request, "sha256": args.expected_request_sha256}
    install_ref = {
        "path": args.release_install_input,
        "sha256": args.expected_release_install_input_sha256,
    }
    _, document = read_exact_request(
        args.workspace_root, request_ref["path"], request_ref["sha256"]
    )
    bootstrap = None
    try:
        if document.get("schema_version") == "cn-daily-production-request.v1":
            bootstrap = BootstrapLaunchInputs(
                workspace=args.workspace_root,
                request_ref=request_ref,
                release_install_ref=install_ref,
            )
            request = bootstrap.request
        else:
            request = validate_automatic_request(document, release_install_ref=install_ref)
    except (ValueError, TypeError) as exc:
        raise CommandError("AUTO_LAUNCH_REQUEST_INVALID") from exc
    if bootstrap is None and request["action"] != "CATCH_UP":
        raise CommandError("AUTO_LAUNCH_CATCHUP_REQUIRED")
    raw, _ = read_exact_request(
        args.workspace_root, args.release_install_input, args.expected_release_install_input_sha256
    )
    try:
        with verified_native_context(
            release_input_raw=raw,
            expected_sha256=install_ref["sha256"],
            repository_root=args.release_repository_root,
        ) as operations:
            yield request_ref, install_ref, request, operations
            if bootstrap is not None:
                bootstrap.recheck()
    except (ValueError, OSError) as exc:
        _evidence_error(exc, automatic=True)


def _inspect(args, request_ref, install_ref, request, operations):
    try:
        value = operations["daily_launch_inspection"](
            workspace=args.workspace_root, request_ref=request_ref, release_install_ref=install_ref
        )
    except (ValueError, OSError) as exc:
        _evidence_error(exc, automatic=True)
    try:
        validate_launch_inspection(
            value,
            request_ref=request_ref,
            release_install_ref=install_ref,
            target_trade_date=request.get("target_trade_date"),
        )
        if value["mode"] == "COMPLETE_READ_ONLY":
            _validate_result(args, value["result"], request_ref, request, install_ref, operations)
    except Exception as exc:
        raise RuntimeError("native launch inspection contract invalid") from exc
    return value


def _inspection_input(args):
    if args.inspection is None or args.expected_inspection_sha256 is None:
        raise CommandError("AUTO_LAUNCH_INSPECTION_REF_REQUIRED")
    path = PurePosixPath(args.inspection)
    prefix = PurePosixPath("data/private/cn_daily_maintenance/launcher_attempts")
    if (
        path.parent.parent != prefix
        or path.name != "inspection.stdout.json"
        or re.fullmatch(r"slot-2020-[0-9]{8}T[0-9]{6}Z-[0-9]+", path.parent.name) is None
    ):
        raise CommandError("AUTO_LAUNCH_INSPECTION_PATH_INVALID")
    return read_exact_request(args.workspace_root, args.inspection, args.expected_inspection_sha256)


def _validate_saved(args, old, actual, request_ref, install_ref, request):
    raw, value = old
    try:
        validate_launch_inspection(
            value,
            request_ref=request_ref,
            release_install_ref=install_ref,
            target_trade_date=request.get("target_trade_date"),
        )
    except (ValueError, TypeError) as exc:
        raise CommandError("AUTO_LAUNCH_INSPECTION_INVALID") from exc
    if raw != canonical_json_bytes(actual) or _inspection_input(args)[0] != raw:
        raise CommandError("AUTO_LAUNCH_INSPECTION_CHANGED")


def _validate_result(args, result, request_ref, request, install_ref, operations):
    if request["action"] == "EXECUTE":
        return _validate_daily_result(
            result, args.workspace_root, request, [request["target_trade_date"]], operations
        )
    return validate_native_automatic_result(
        result, args.workspace_root, request_ref, request, install_ref, operations
    )


def _recover(args, actual, request_ref, request, install_ref, operations):
    if actual["mode"] != "LOCAL_REPAIR":
        raise CommandError("DAILY_LAUNCH_RECOVERY_REQUIRES_LOCAL_SCOPE")
    restriction = (
        {"committed_recovery_only": True}
        if actual["schema_version"] == BOOTSTRAP_SCHEMA
        or (
            actual["schema_version"] == REGISTERED_SCHEMA
            and actual["recovery_scope"] == "COMMITTED_DAG_RECOVERY"
        )
        else {"no_producers": True}
    )
    result = operations["daily_close"](
        workspace=args.workspace_root,
        request_ref=request_ref,
        release_install_ref=install_ref,
        **restriction,
    )
    _validate_result(args, result, request_ref, request, install_ref, operations)
    sys.stdout.buffer.write(canonical_json_bytes(result))
    return (
        production_result_exit_code
        if request["action"] == "EXECUTE"
        else automatic_result_exit_code
    )(result)


def _dispatch(argv):
    parser = MachineArgumentParser(prog="installed-daily-launch")
    parser.add_argument("--mode", choices=("inspect", "validate", "emit", "recover"), required=True)
    for name in (
        "workspace-root",
        "request",
        "expected-request-sha256",
        "release-repository-root",
        "release-install-input",
        "expected-release-install-input-sha256",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--inspection")
    parser.add_argument("--expected-inspection-sha256")
    args = parser.parse_args(argv)
    if args.mode == "inspect" and (
        args.inspection is not None or args.expected_inspection_sha256 is not None
    ):
        raise CommandError("AUTO_LAUNCH_INSPECTION_ARGUMENTS_INVALID")
    old = None if args.mode == "inspect" else _inspection_input(args)
    with _installed(args) as (request_ref, install_ref, request, operations):
        actual = _inspect(args, request_ref, install_ref, request, operations)
        if old is not None:
            _validate_saved(args, old, actual, request_ref, install_ref, request)
        if args.mode == "recover":
            return _recover(args, actual, request_ref, request, install_ref, operations)
        if args.mode == "emit":
            if actual["mode"] != "COMPLETE_READ_ONLY":
                raise CommandError("AUTO_LAUNCH_EMIT_REQUIRES_COMPLETE")
            sys.stdout.buffer.write(canonical_json_bytes(actual["result"]))
            return 0
        if args.mode == "inspect":
            sys.stdout.buffer.write(canonical_json_bytes(actual))
        return MODES[actual["mode"]]


def main(argv=None):
    return command_boundary(lambda: _dispatch(sys.argv[1:] if argv is None else argv))
