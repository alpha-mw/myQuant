"""Fixed installed source modes for the existing daily slot launcher."""

from pathlib import PurePosixPath
import re
import sys

from quant_investor.contracts import canonical_json_bytes
from quant_investor.cli.input import read_exact_request
from quant_investor.cli.output import CommandError, MachineArgumentParser, command_boundary
from quant_investor.operations.native_bridge import verified_native_context
from quant_investor.operations.daily_preparation_contract import PreparationError, validate_config
from quant_investor.operations.source_slot_contract import SourceSlotError, MODES, validate_result
from quant_investor.market.cn_benchmark_store import CNBenchmarkStoreError
from quant_investor.market.close_session_authority import CloseSessionAuthorityError
from quant_investor.market.tushare_transport import TushareHttpsError
from quant_investor.strategy_records.event_store import StrategyEventStoreError
from quant_investor.strategy_records.store import StrategyRecordStoreError


def _saved(args):
    if args.inspection is None or args.expected_inspection_sha256 is None:
        raise CommandError("SOURCE_SAVED_INSPECTION_REQUIRED")
    path = PurePosixPath(args.inspection)
    if (
        str(path.parent.parent) != "data/private/cn_daily_maintenance/launcher_attempts"
        or re.fullmatch(r"slot-2020-[0-9]{8}T[0-9]{6}Z-[0-9]+", path.parent.name) is None
        or path.name not in {"source-inspection.stdout.json", "source-provision.stdout.json"}
    ):
        raise CommandError("SOURCE_SAVED_INSPECTION_PATH_INVALID")
    return read_exact_request(args.workspace_root, args.inspection, args.expected_inspection_sha256)


def _output(args, result, old):
    if old is not None:
        if old[0] != canonical_json_bytes(result) or _saved(args)[0] != old[0]:
            raise CommandError("SOURCE_INSPECTION_CHANGED")
    if args.mode == "select":
        if result["mode"] == "NON_TRADING_DAY":
            return 10  # Internal selection result; shell emits validated JSON and exits0.
        if result["mode"] != "REQUEST_AVAILABLE":
            raise CommandError("SOURCE_REQUEST_NOT_AVAILABLE")
        ref = result["request_ref"]
        if (
            not ref["path"]
            or not all(32 <= ord(c) <= 126 for c in ref["path"])
            or re.fullmatch("[a-f0-9]{64}", ref["sha256"]) is None
        ):
            raise CommandError("SOURCE_REQUEST_TRANSPORT_INVALID")
        sys.stdout.write(ref["path"] + "\n" + ref["sha256"] + "\n")
        return 0
    if args.mode in {"provision", "emit"} and result["mode"] not in {
        "REQUEST_AVAILABLE",
        "NON_TRADING_DAY",
    }:
        raise CommandError("SOURCE_PROVISION_INCOMPLETE")
    sys.stdout.buffer.write(canonical_json_bytes(result))
    return MODES[result["mode"]]


def _dispatch(argv):
    parser = MachineArgumentParser(prog="installed-daily-sources")
    parser.add_argument("--mode", choices=("inspect", "provision", "emit", "select"), required=True)
    for name in (
        "workspace-root",
        "config",
        "expected-config-sha256",
        "release-repository-root",
        "release-install-input",
        "expected-release-install-input-sha256",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--no-providers", action="store_true")
    parser.add_argument("--inspection")
    parser.add_argument("--expected-inspection-sha256")
    args = parser.parse_args(argv)
    if args.mode in {"inspect", "provision"} and (
        args.inspection is not None or args.expected_inspection_sha256 is not None
    ):
        raise CommandError("SOURCE_INSPECTION_ARGUMENTS_INVALID")
    old = _saved(args) if args.mode in {"emit", "select"} else None
    cfg_ref = {"path": args.config, "sha256": args.expected_config_sha256}
    install_ref = {
        "path": args.release_install_input,
        "sha256": args.expected_release_install_input_sha256,
    }
    cfg_raw, cfg = read_exact_request(args.workspace_root, cfg_ref["path"], cfg_ref["sha256"])
    raw, _ = read_exact_request(args.workspace_root, install_ref["path"], install_ref["sha256"])
    try:
        config = validate_config(cfg)
        if config["release_install_ref"] != install_ref:
            raise SourceSlotError("SOURCE_CONFIG_INSTALL_MISMATCH")
        with verified_native_context(
            release_input_raw=raw,
            expected_sha256=install_ref["sha256"],
            repository_root=args.release_repository_root,
        ) as operations:
            result = operations["daily_source_inputs"](
                workspace=args.workspace_root,
                config_ref=cfg_ref,
                release_install_ref=install_ref,
                mode="provision" if args.mode == "provision" else "inspect",
                no_providers=args.no_providers,
            )
            validate_result(result, config_ref=cfg_ref, release_install_ref=install_ref)
            if (
                read_exact_request(args.workspace_root, cfg_ref["path"], cfg_ref["sha256"])[0]
                != cfg_raw
            ):
                raise SourceSlotError("SOURCE_CONFIG_CHANGED")
            return _output(args, result, old)
    except (SourceSlotError, PreparationError) as exc:
        raise CommandError(exc.code, fields=exc.fields) from exc
    except (
        ValueError,
        FileNotFoundError,
        CNBenchmarkStoreError,
        CloseSessionAuthorityError,
        TushareHttpsError,
        StrategyEventStoreError,
        StrategyRecordStoreError,
    ) as exc:
        raise CommandError("SOURCE_NATIVE_INPUT_REJECTED") from exc


def main(argv=None):
    return command_boundary(lambda: _dispatch(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
