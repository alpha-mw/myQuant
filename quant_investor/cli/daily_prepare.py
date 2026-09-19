"""Installed, local-only input preparation entrypoint for the existing launcher."""

import sys

from quant_investor.contracts import canonical_json_bytes
from quant_investor.cli.input import read_exact_request
from quant_investor.cli.output import CommandError, MachineArgumentParser, command_boundary
from quant_investor.operations.daily_preparation_contract import PreparationError, validate_config
from quant_investor.operations.daily_preparation import prepare_daily_request
from quant_investor.system.release_install import verify_running_release_install_input
from quant_investor.system.errors import SystemStorageError


def _dispatch(argv):
    parser = MachineArgumentParser(prog="installed-daily-prepare")
    for name in (
        "workspace-root",
        "config",
        "expected-config-sha256",
        "calendar",
        "expected-calendar-sha256",
        "raw-calendar",
        "expected-raw-calendar-sha256",
        "release-repository-root",
    ):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    config_ref = {"path": args.config, "sha256": args.expected_config_sha256}
    raw, config = read_exact_request(args.workspace_root, config_ref["path"], config_ref["sha256"])
    try:
        config = validate_config(config)
        install = config["release_install_ref"]
        installed_raw, _ = read_exact_request(
            args.workspace_root, install["path"], install["sha256"]
        )
        verify_running_release_install_input(
            installed_raw, repository_root=args.release_repository_root
        )
        result = prepare_daily_request(
            workspace=args.workspace_root,
            config_ref=config_ref,
            calendar_ref={"path": args.calendar, "sha256": args.expected_calendar_sha256},
            raw_calendar_ref={
                "path": args.raw_calendar,
                "sha256": args.expected_raw_calendar_sha256,
            },
        )
        if (
            read_exact_request(args.workspace_root, config_ref["path"], config_ref["sha256"])[0]
            != raw
        ):
            raise PreparationError("PREPARATION_CONFIG_CHANGED")
    except PreparationError as exc:
        raise CommandError(exc.code, fields=exc.fields) from exc
    except (ValueError, OSError, SystemStorageError) as exc:
        raise CommandError("PREPARATION_INPUT_REJECTED") from exc
    sys.stdout.buffer.write(canonical_json_bytes(result))
    return 0


def main(argv=None):
    return command_boundary(lambda: _dispatch(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
