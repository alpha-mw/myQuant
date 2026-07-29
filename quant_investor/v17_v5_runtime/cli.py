"""Non-operational V17 v5 Phase-0 verification CLI."""

from __future__ import annotations

import argparse
import json
from typing import Any, Sequence

from quant_investor.v17_v5_contract import (
    PROTOCOL_VERSION,
    verify_package,
    verify_predecessor,
    verify_runtime_build,
)
from quant_investor.v17_v5_contract.resources import PackageResourceError

from .authority import (
    DELIVERY_STATUS,
    GLOBAL_ACTIVATION_STATE,
    RUN_STATE,
    STATE,
    authority_envelope,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quant-investor-v17-v5")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="show the inert V17 v5 authority boundary")
    commands.add_parser("verify", help="verify v5 package/runtime and the pinned v4 predecessor")
    return parser


def _wire() -> dict[str, Any]:
    authority = authority_envelope()
    return {
        **authority,
        "default_protocol_state": STATE,
        "global_activation_state": GLOBAL_ACTIVATION_STATE,
        "protocol_version": PROTOCOL_VERSION,
        "run_state": RUN_STATE,
        "state": STATE,
        "status": DELIVERY_STATUS,
        "version": "myquant.v17.v5.phase0-status.v1",
    }


def _emit(payload: Any) -> None:
    print(
        json.dumps(
            payload, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "status":
        _emit(_wire())
        return 0
    try:
        package = verify_package()
        runtime = verify_runtime_build()
        predecessor = verify_predecessor()
    except PackageResourceError as exc:
        _emit({**_wire(), "error": str(exc), "verified": False})
        return exc.exit_code
    _emit(
        {
            **_wire(),
            "package_asset_count": len(package),
            "predecessor": predecessor,
            "runtime_source_count": len(runtime),
            "verified": True,
        }
    )
    return 0


__all__ = ["main"]
