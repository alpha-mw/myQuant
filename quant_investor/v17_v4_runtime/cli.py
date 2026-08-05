"""Explicit research-only V17 v4 forward-evidence and Shadow CLI."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from quant_investor.v17_v4_contract import verify_package

from .deep_v3 import compile_deep_v3
from .forward_shadow import (
    build_shadow_readiness_v2,
    read_forward_shadow_session,
)
from .orchestrator import ForwardEvidenceError, run_forward
from .research_factor_set import ResearchFactorSetStore


_NO_AUTHORITY = {
    "broker": False,
    "execution": False,
    "mainline_authority": False,
    "order": False,
    "production": False,
    "research_only": True,
    "trade": False,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quant-investor-v17-v4",
        description=(
            "explicit research-only V17 v4 forward-evidence and Shadow surface; "
            "no provider, production, execution, broker, order, or trade authority"
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify", help="verify packaged V17 v4 contract bytes")
    commands.add_parser("status", help="show the fail-closed public-surface state")
    run_forward_parser = commands.add_parser(
        "run-forward",
        help="run one immutable EXPLORE or FORWARD_EVIDENCE request",
    )
    run_forward_parser.add_argument(
        "--workspace-root",
        default=str(Path.cwd()),
    )
    run_forward_parser.add_argument("--request-path", required=True)
    run_forward_parser.add_argument("--request-sha256", required=True)
    factor_set_status = commands.add_parser(
        "factor-set-status",
        help="read the exact current monthly rotating research factor set",
    )
    factor_set_status.add_argument(
        "--workspace-root",
        default=str(Path.cwd()),
    )
    deep_v3_compile = commands.add_parser(
        "deep-v3-compile",
        help="compile owner-prepositioned Deep v3 evidence for Fusion v2",
    )
    deep_v3_compile.add_argument("--workspace-root", default=str(Path.cwd()))
    deep_v3_compile.add_argument("--assessment-manifest-path", required=True)
    deep_v3_compile.add_argument(
        "--assessment-manifest-sha256",
        required=True,
    )
    deep_v3_compile.add_argument("--created-at", required=True)
    forward_readiness = commands.add_parser(
        "forward-shadow-readiness",
        help="publish blocker-only dynamic Shadow readiness without model output",
    )
    forward_readiness.add_argument(
        "--workspace-root",
        default=str(Path.cwd()),
    )
    forward_readiness.add_argument("--readiness-id", required=True)
    forward_readiness.add_argument("--strategy-id", required=True)
    forward_readiness.add_argument("--cutoff", required=True)
    forward_readiness.add_argument("--decision-session", required=True)
    forward_readiness.add_argument("--created-at", required=True)
    forward_readiness.add_argument(
        "--blocker-code",
        action="append",
        required=True,
    )
    forward_readiness.add_argument(
        "--factor-refs-present",
        action="store_true",
    )
    forward_status = commands.add_parser(
        "forward-shadow-status",
        help="read and replay one exact dynamic Shadow v3 session ref",
    )
    forward_status.add_argument(
        "--workspace-root",
        default=str(Path.cwd()),
    )
    forward_status.add_argument("--session-ref-path", required=True)
    forward_status.add_argument("--session-ref-sha256", required=True)
    return parser


def _wire(*, package_assets: int | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": "myquant.v17.v4.scaffold-status.v1",
        "status": "RESEARCH_ONLY",
        "authority": dict(_NO_AUTHORITY),
        "research_only": True,
        "provider_calls": False,
        "llm_control_calls": False,
        "execution_calls": False,
        "broker_calls": False,
        "order_calls": False,
        "trade_calls": False,
        "mainline_authority": False,
        "production_authority": False,
        "run_state": "INACTIVE",
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
    if args.command == "run-forward":
        try:
            _emit(
                run_forward(
                    str(Path(args.workspace_root).resolve()),
                    request_path=args.request_path,
                    request_sha256=args.request_sha256,
                )
            )
            return 0
        except ForwardEvidenceError as exc:
            _emit(
                {
                    "authority": dict(_NO_AUTHORITY),
                    "broker": False,
                    "blocker_code": exc.code,
                    "execution": False,
                    "mainline_authority": False,
                    "order": False,
                    "production_authority": False,
                    "research_only": True,
                    "run_state": exc.run_state,
                    "side_effects": {
                        "broker": False,
                        "execution": False,
                        "order": False,
                        "provider": False,
                        "trade": False,
                    },
                    "status": "BLOCKED",
                    "trade": False,
                }
            )
            return exc.exit_code
    if args.command == "factor-set-status":
        state = ResearchFactorSetStore(
            str(Path(args.workspace_root).resolve()),
        ).read_current()
        _emit(
            {
                "factor_set": state.factor_set,
                "factor_set_ref": state.factor_set_ref,
                "pointer": state.pointer,
                "pointer_ref": state.pointer_ref,
                "status": "CURRENT_RESEARCH_FACTOR_SET_READY",
            }
        )
        return 0
    if args.command == "deep-v3-compile":
        _emit(
            compile_deep_v3(
                str(Path(args.workspace_root).resolve()),
                assessment_manifest_path=args.assessment_manifest_path,
                expected_assessment_manifest_sha256=(args.assessment_manifest_sha256),
                created_at=args.created_at,
            )
        )
        return 0
    if args.command == "forward-shadow-readiness":
        _emit(
            build_shadow_readiness_v2(
                str(Path(args.workspace_root).resolve()),
                readiness_id=args.readiness_id,
                strategy_id=args.strategy_id,
                cutoff=args.cutoff,
                decision_session=args.decision_session,
                created_at=args.created_at,
                blocker_codes=args.blocker_code,
                factor_refs_present=args.factor_refs_present,
            )
        )
        return 0
    if args.command == "forward-shadow-status":
        _emit(
            read_forward_shadow_session(
                str(Path(args.workspace_root).resolve()),
                session_ref_path=args.session_ref_path,
                expected_session_ref_sha256=args.session_ref_sha256,
            )
        )
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
