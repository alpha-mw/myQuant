"""Explicit no-write CLI for the V17 v4 contract scaffold."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from quant_investor.v17_v4_contract import verify_package

from .authority import DELIVERY_STATUS, authority_envelope


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quant-investor-v17-v4",
        description=(
            "explicit V17 v4 production-research contract scaffold; "
            "no activation, default-selector, provider, execution, broker, order, or trade authority"
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify", help="verify packaged V17 v4 contract bytes")
    commands.add_parser("status", help="show the fail-closed scaffold state")
    return parser


def _wire(*, package_assets: int | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": "myquant.v17.v4.scaffold-status.v1",
        "status": DELIVERY_STATUS,
        **authority_envelope(),
        "provider_calls": False,
        "llm_control_calls": False,
        "execution_calls": False,
        "broker_calls": False,
        "order_calls": False,
        "trade_calls": False,
        "selector_writes": False,
    }
    if package_assets is not None:
        payload["package_verified"] = True
        payload["package_asset_count"] = package_assets
    return payload


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify":
        _emit(_wire(package_assets=len(verify_package())))
        return 0
    if args.command == "status":
        _emit(_wire())
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
