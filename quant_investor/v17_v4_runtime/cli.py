"""Explicit V17 v4 verification, read-only, and canary-only CLI."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from quant_investor.v17_v4_contract import verify_package

from .authority import DELIVERY_STATUS, authority_envelope
from .public_surfaces import (
    build_dashboard_contract_v4,
    build_public_surface_compatibility_receipts,
    publish_canary_snapshot,
    resolve_public_run,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quant-investor-v17-v4",
        description=(
            "explicit V17 v4 production-research surface; "
            "no default-selector, provider, execution, broker, order, or trade authority"
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify", help="verify packaged V17 v4 contract bytes")
    commands.add_parser("status", help="show the fail-closed public-surface state")
    read_formal = commands.add_parser(
        "read-formal",
        help="read one exact FORMAL_ACTIVE run as an explicit CLI canary",
    )
    read_formal.add_argument("--workspace-root", default=str(Path.cwd()))
    read_formal.add_argument("--strategy-id", required=True)
    dashboard = commands.add_parser(
        "dashboard",
        help="emit Dashboard Contract v4 for one FORMAL_ACTIVE canary run",
    )
    dashboard.add_argument("--workspace-root", default=str(Path.cwd()))
    dashboard.add_argument("--strategy-id", required=True)
    audit = commands.add_parser(
        "audit-surfaces",
        help="emit four hash-bound public-surface compatibility receipts",
    )
    audit.add_argument("--repo-root", default=str(Path.cwd()))
    audit.add_argument("--workspace-root", default=str(Path.cwd()))
    audit.add_argument("--strategy-id", required=True)
    audit.add_argument("--created-at", required=True)
    publish = commands.add_parser(
        "publish-canary",
        help="write one immutable schedule snapshot under results/v17_v4_canary",
    )
    publish.add_argument("--workspace-root", default=str(Path.cwd()))
    publish.add_argument("--strategy-id", required=True)
    publish.add_argument("--session-id", required=True)
    publish.add_argument("--created-at", required=True)
    publish.add_argument(
        "--expected-formal-pointer-sha256",
        required=True,
    )
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
    if args.command == "read-formal":
        _emit(
            resolve_public_run(
                Path(args.workspace_root),
                strategy_id=args.strategy_id,
                surface="CLI",
            )
        )
        return 0
    if args.command == "dashboard":
        _emit(
            build_dashboard_contract_v4(
                Path(args.workspace_root),
                strategy_id=args.strategy_id,
            )
        )
        return 0
    if args.command == "audit-surfaces":
        receipts = build_public_surface_compatibility_receipts(
            Path(args.repo_root),
            Path(args.workspace_root),
            strategy_id=args.strategy_id,
            created_at=args.created_at,
        )
        _emit(
            {
                "protocol_version": "myquant.v17.v4",
                "receipt_count": len(receipts),
                "receipts": list(receipts),
                "status": "ACCEPTED",
            }
        )
        return 0
    if args.command == "publish-canary":
        _emit(
            publish_canary_snapshot(
                Path(args.workspace_root),
                strategy_id=args.strategy_id,
                session_id=args.session_id,
                created_at=args.created_at,
                expected_formal_pointer_sha256=(
                    args.expected_formal_pointer_sha256
                ),
            )
        )
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
