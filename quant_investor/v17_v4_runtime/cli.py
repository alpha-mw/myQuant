"""Explicit V17 v4 verification, Shadow research, and canary-only CLI."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from quant_investor.v17_v4_contract import verify_package

from .authority import DELIVERY_STATUS, authority_envelope
from .deep_v2 import compile_deep_v2
from .deep_v3 import compile_deep_v3
from .forward_shadow import (
    build_shadow_readiness_v2,
    read_forward_shadow_session,
)
from .orchestrator import ForwardEvidenceError, run_forward
from .public_surfaces import (
    build_dashboard_contract_v4,
    build_public_surface_compatibility_receipts,
    publish_canary_snapshot,
    resolve_public_run,
)
from .research_quant import compile_research_quant_branch
from .research_factor_set import ResearchFactorSetStore
from .shadow_runtime import publish_shadow_run, read_shadow_session


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
    run_forward_parser = commands.add_parser(
        "run-forward",
        help=("run one immutable EXPLORE, FORWARD_EVIDENCE, or " "RELEASE_CANDIDATE request"),
    )
    run_forward_parser.add_argument(
        "--workspace-root",
        default=str(Path.cwd()),
    )
    run_forward_parser.add_argument("--request-path", required=True)
    run_forward_parser.add_argument("--request-sha256", required=True)
    deep_compile = commands.add_parser(
        "deep-compile",
        help="compile prepositioned official Deep evidence into a shadow-only v2 bundle",
    )
    deep_compile.add_argument("--workspace-root", default=str(Path.cwd()))
    deep_compile.add_argument(
        "--assessment-manifest-path",
        required=True,
    )
    deep_compile.add_argument(
        "--assessment-manifest-sha256",
        required=True,
    )
    deep_compile.add_argument("--created-at", required=True)
    quant_compile = commands.add_parser(
        "quant-compile",
        help=("replay the legacy fixed-trio V17 v4 research Quant branch"),
    )
    quant_compile.add_argument(
        "--workspace-root",
        default=str(Path.cwd()),
    )
    quant_compile.add_argument("--run-id", required=True)
    quant_compile.add_argument("--output-id", required=True)
    quant_compile.add_argument("--initial-pool-path", required=True)
    quant_compile.add_argument("--initial-pool-sha256", required=True)
    quant_compile.add_argument("--market-slice-path", required=True)
    quant_compile.add_argument("--market-slice-sha256", required=True)
    shadow_publish = commands.add_parser(
        "shadow-publish",
        help="publish a Factor-v4-gated immutable shadow run or blocker readiness",
    )
    shadow_publish.add_argument(
        "--workspace-root",
        default=str(Path.cwd()),
    )
    shadow_publish.add_argument("--readiness-id", required=True)
    shadow_publish.add_argument("--shadow-run-id", required=True)
    shadow_publish.add_argument("--strategy-id", required=True)
    shadow_publish.add_argument("--cutoff", required=True)
    shadow_publish.add_argument("--decision-session", required=True)
    shadow_publish.add_argument("--created-at", required=True)
    for name in (
        "factor-active-set",
        "factor-control-receipt",
        "source-locator",
        "initial-pool",
        "quant-branch",
        "fundamental-branch",
        "fusion-top24",
        "deep-bundle",
        "holdings-snapshot",
    ):
        shadow_publish.add_argument(f"--{name}-path")
        shadow_publish.add_argument(f"--{name}-sha256")
    shadow_publish.add_argument(
        "--research-factor-shadow-only-override-id",
        help=(
            "legacy default-off assertion ID for one exact fixed-trio "
            "Shadow replay; cannot coexist with formal Factor refs"
        ),
    )
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
    shadow_status = commands.add_parser(
        "shadow-status",
        help="read one exact immutable v4 shadow session",
    )
    shadow_status.add_argument(
        "--workspace-root",
        default=str(Path.cwd()),
    )
    shadow_status.add_argument("--strategy-id", required=True)
    shadow_status.add_argument("--decision-session", required=True)
    shadow_status.add_argument("--expected-sha256")
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
        "default_protocol_state": "V15_DEFAULT",
        "formal_activation_eligible": False,
        "global_activation_state": "INACTIVE",
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
                    "authority": False,
                    "broker": False,
                    "blocker_code": exc.code,
                    "execution": False,
                    "formal_activation_eligible": False,
                    "global_activation_state": exc.global_activation_state,
                    "order": False,
                    "research_runtime_default": False,
                    "run_state": exc.run_state,
                    "side_effects": {
                        "broker": False,
                        "canary": False,
                        "default": False,
                        "execution": False,
                        "formal": False,
                        "order": False,
                        "promotion": False,
                        "provider": False,
                        "selector": False,
                        "trade": False,
                    },
                    "status": "BLOCKED",
                    "trade": False,
                }
            )
            return exc.exit_code
    if args.command == "deep-compile":
        _emit(
            compile_deep_v2(
                args.workspace_root,
                assessment_manifest_path=(args.assessment_manifest_path),
                expected_assessment_manifest_sha256=(args.assessment_manifest_sha256),
                created_at=args.created_at,
            )
        )
        return 0
    if args.command == "quant-compile":
        _emit(
            compile_research_quant_branch(
                args.workspace_root,
                run_id=args.run_id,
                output_id=args.output_id,
                initial_pool_path=args.initial_pool_path,
                initial_pool_sha256=args.initial_pool_sha256,
                market_slice_path=args.market_slice_path,
                market_slice_sha256=args.market_slice_sha256,
            )
        )
        return 0
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
    if args.command == "shadow-publish":
        _emit(
            publish_shadow_run(
                args.workspace_root,
                readiness_id=args.readiness_id,
                shadow_run_id=args.shadow_run_id,
                strategy_id=args.strategy_id,
                cutoff=args.cutoff,
                decision_session=args.decision_session,
                created_at=args.created_at,
                factor_active_set_path=args.factor_active_set_path,
                factor_active_set_sha256=(args.factor_active_set_sha256),
                factor_control_receipt_path=(args.factor_control_receipt_path),
                factor_control_receipt_sha256=(args.factor_control_receipt_sha256),
                research_factor_shadow_only_override_id=(
                    args.research_factor_shadow_only_override_id
                ),
                source_locator_path=args.source_locator_path,
                source_locator_sha256=args.source_locator_sha256,
                initial_pool_path=args.initial_pool_path,
                initial_pool_sha256=args.initial_pool_sha256,
                quant_branch_path=args.quant_branch_path,
                quant_branch_sha256=args.quant_branch_sha256,
                fundamental_branch_path=(args.fundamental_branch_path),
                fundamental_branch_sha256=(args.fundamental_branch_sha256),
                fusion_top24_path=args.fusion_top24_path,
                fusion_top24_sha256=args.fusion_top24_sha256,
                deep_bundle_path=args.deep_bundle_path,
                deep_bundle_sha256=args.deep_bundle_sha256,
                holdings_snapshot_path=args.holdings_snapshot_path,
                holdings_snapshot_sha256=(args.holdings_snapshot_sha256),
            )
        )
        return 0
    if args.command == "shadow-status":
        _emit(
            read_shadow_session(
                args.workspace_root,
                strategy_id=args.strategy_id,
                decision_session=args.decision_session,
                expected_sha256=args.expected_sha256,
            )
        )
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
                expected_formal_pointer_sha256=(args.expected_formal_pointer_sha256),
            )
        )
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
