"""Explicit V17 v4 verification, Shadow research, and canary-only CLI."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Final

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
from .regime_evidence_v2 import (
    RegimeEvidenceV2Error,
    RegimeEvidenceV2InputGap,
    build_regime_evidence_v2,
    read_regime_evidence_v2,
)
from .shadow_runtime import publish_shadow_run, read_shadow_session
from .source_builder import (
    SourceSnapshotError,
    SourceSnapshotGap,
    build_source_snapshot,
    gap_payload,
)

REGIME_EVIDENCE_AUTHORITY_ATTESTATION: Final = {
    "broker": False,
    "execution": False,
    "formal_research_publication": False,
    "order": False,
    "research_runtime_default": False,
    "trade": False,
}

REGIME_EVIDENCE_SIDE_EFFECTS: Final = {
    "broker_calls": False,
    "execution_calls": False,
    "factor_governance_writes": False,
    "order_calls": False,
    "portfolio_writes": False,
    "provider_calls": False,
    "selector_writes": False,
    "trade_calls": False,
}


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
    build_source = commands.add_parser(
        "build-source-snapshot",
        help=(
            "materialize one immutable, offline Forward Evidence source "
            "snapshot from exact canonical inputs"
        ),
    )
    build_source.add_argument("--workspace-root", default=str(Path.cwd()))
    build_source.add_argument("--strategy-id", required=True)
    build_source.add_argument("--decision-session", required=True)
    build_source.add_argument("--cutoff", required=True)
    build_source.add_argument("--market-pointer-sha256", required=True)
    build_source.add_argument("--fundamental-pointer-sha256", required=True)
    build_source.add_argument("--factor-set-pointer-sha256", required=True)
    build_source.add_argument("--strategy-universe-path", required=True)
    build_source.add_argument("--strategy-universe-sha256", required=True)
    build_source.add_argument(
        "--strategy-universe-manifest-path",
        required=True,
    )
    build_source.add_argument(
        "--strategy-universe-manifest-sha256",
        required=True,
    )
    regime_evidence_build = commands.add_parser(
        "regime-evidence-build",
        help=(
            "build one immutable, filtered-causal Regime Evidence v2 "
            "artifact from explicit local path and SHA closure"
        ),
    )
    regime_evidence_build.add_argument("--workspace-root", required=True)
    regime_evidence_build.add_argument("--evidence-id", required=True)
    regime_evidence_build.add_argument("--strategy-id", required=True)
    regime_evidence_build.add_argument("--decision-session", required=True)
    regime_evidence_build.add_argument("--cutoff", required=True)
    regime_evidence_build.add_argument("--created-at", required=True)
    for name in (
        "inference-policy",
        "model-snapshot",
        "transition-matrix",
        "feature-snapshot",
    ):
        regime_evidence_build.add_argument(f"--{name}-path", required=True)
        regime_evidence_build.add_argument(f"--{name}-sha256", required=True)
    regime_evidence_build.add_argument("--prior-evidence-path")
    regime_evidence_build.add_argument("--prior-evidence-sha256")
    regime_evidence_status = commands.add_parser(
        "regime-evidence-status",
        help="read and replay one exact Regime Evidence v2 artifact",
    )
    regime_evidence_status.add_argument("--workspace-root", required=True)
    regime_evidence_status.add_argument(
        "--artifact-path",
        "--evidence-path",
        dest="artifact_path",
        required=True,
    )
    regime_evidence_status.add_argument(
        "--expected-sha256",
        "--evidence-sha256",
        dest="expected_sha256",
        required=True,
    )
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


def _regime_evidence_base_payload() -> dict[str, Any]:
    return {
        "artifact_version": "myquant.v17.v4.regime-evidence.v2",
        "authority": dict(REGIME_EVIDENCE_AUTHORITY_ATTESTATION),
        "broker": False,
        "default_protocol_state": "V15_DEFAULT",
        "execution": False,
        "factor_governance_write": False,
        "formal_activation": False,
        "global_activation_state": "INACTIVE",
        "order": False,
        "promotion": False,
        "protocol_version": "myquant.v17.v4",
        "provider_calls": False,
        "research_runtime_default": False,
        "run_state": "INACTIVE",
        "selector": False,
        "side_effects": dict(REGIME_EVIDENCE_SIDE_EFFECTS),
        "trade": False,
    }


def _mapping(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value) and not isinstance(value, type):
        converted = asdict(value)
        if isinstance(converted, dict):
            return converted
    raise TypeError("Regime Evidence v2 result must be a mapping or dataclass")


def _regime_evidence_success_payload(
    value: object,
    *,
    status_read: bool,
) -> dict[str, Any]:
    result = _mapping(value)
    document_value = result.get("document", result)
    document = _mapping(document_value)
    payload = _regime_evidence_base_payload()
    for key in (
        "available_at",
        "created",
        "created_at",
        "coverage_ratio",
        "cutoff",
        "decision_session",
        "effective_session",
        "evidence_id",
        "evidence_path",
        "evidence_sha256",
        "hard_state",
        "inference_kind",
        "model_id",
        "model_version",
        "market_sample_count",
        "minimum_market_sample",
        "observed_through_session",
        "publication_phase",
        "published_at",
        "reused",
        "scope_kind",
        "smoothing_used",
        "state_probabilities",
        "strategy_id",
    ):
        if key in result:
            payload[key] = result[key]
        elif key in document:
            payload[key] = document[key]
    payload["blocker_codes"] = list(
        document.get(
            "blocker_codes",
            result.get("blocker_codes", []),
        )
        or []
    )
    payload["evidence"] = document
    payload["replay_result"] = document.get(
        "replay_result",
        result.get(
            "replay_result",
            {
                "closure_replayed": True,
                "hard_state_reclassified": False,
                "status": "EXACT_REPLAY_VERIFIED",
            },
        ),
    )
    payload["status"] = (
        "AVAILABLE"
        if status_read
        else str(result.get("status") or document.get("status") or "AVAILABLE")
    )
    return payload


def _regime_evidence_failure_payload(
    *,
    status: str,
    blocker_code: str,
    detail: str,
) -> dict[str, Any]:
    return {
        **_regime_evidence_base_payload(),
        "blocker_codes": [blocker_code],
        "detail": detail,
        "replay_result": {
            "closure_replayed": False,
            "hard_state_reclassified": False,
            "status": "NOT_AVAILABLE",
        },
        "status": status,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify":
        _emit(_wire(package_assets=len(verify_package())))
        return 0
    if args.command == "status":
        _emit(_wire())
        return 0
    if args.command == "build-source-snapshot":
        try:
            _emit(
                build_source_snapshot(
                    str(Path(args.workspace_root).resolve()),
                    strategy_id=args.strategy_id,
                    decision_session=args.decision_session,
                    cutoff=args.cutoff,
                    market_pointer_sha256=args.market_pointer_sha256,
                    fundamental_pointer_sha256=(args.fundamental_pointer_sha256),
                    factor_set_pointer_sha256=(args.factor_set_pointer_sha256),
                    strategy_universe_path=args.strategy_universe_path,
                    strategy_universe_sha256=(args.strategy_universe_sha256),
                    strategy_universe_manifest_path=(args.strategy_universe_manifest_path),
                    strategy_universe_manifest_sha256=(args.strategy_universe_manifest_sha256),
                )
            )
            return 0
        except SourceSnapshotGap as exc:
            _emit(gap_payload(exc))
            return exc.exit_code
        except SourceSnapshotError as exc:
            gap = SourceSnapshotGap(f"SOURCE_SNAPSHOT_BUILD_FAILED: {exc}")
            _emit(gap_payload(gap))
            return gap.exit_code
    if args.command == "regime-evidence-build":
        if bool(args.prior_evidence_path) != bool(args.prior_evidence_sha256):
            _emit(
                _regime_evidence_failure_payload(
                    status="BLOCKED",
                    blocker_code="PRIOR_EVIDENCE_EXPLICIT_PAIR_REQUIRED",
                    detail=(
                        "--prior-evidence-path and --prior-evidence-sha256 "
                        "must be supplied together"
                    ),
                )
            )
            return 2
        try:
            build_result = build_regime_evidence_v2(
                workspace_root=str(Path(args.workspace_root).resolve()),
                evidence_id=args.evidence_id,
                strategy_id=args.strategy_id,
                decision_session=args.decision_session,
                cutoff=args.cutoff,
                created_at=args.created_at,
                inference_policy_path=args.inference_policy_path,
                inference_policy_sha256=args.inference_policy_sha256,
                model_snapshot_path=args.model_snapshot_path,
                model_snapshot_sha256=args.model_snapshot_sha256,
                transition_matrix_path=args.transition_matrix_path,
                transition_matrix_sha256=args.transition_matrix_sha256,
                feature_snapshot_path=args.feature_snapshot_path,
                feature_snapshot_sha256=args.feature_snapshot_sha256,
                prior_evidence_path=args.prior_evidence_path,
                prior_evidence_sha256=args.prior_evidence_sha256,
            )
            _emit(
                _regime_evidence_success_payload(
                    build_result,
                    status_read=False,
                )
            )
            return 0
        except RegimeEvidenceV2InputGap as exc:
            code = str(
                getattr(
                    exc,
                    "blocker_code",
                    "TRUE_CURRENT_CANONICAL_INPUT_GAP",
                )
            )
            _emit(
                _regime_evidence_failure_payload(
                    status="TRUE_CURRENT_CANONICAL_INPUT_GAP",
                    blocker_code=code,
                    detail=str(exc),
                )
            )
            return 2
        except RegimeEvidenceV2Error as exc:
            code = str(getattr(exc, "blocker_code", "REGIME_EVIDENCE_V2_BLOCKED"))
            _emit(
                _regime_evidence_failure_payload(
                    status="BLOCKED",
                    blocker_code=code,
                    detail=str(exc),
                )
            )
            return 2
    if args.command == "regime-evidence-status":
        try:
            status_result = read_regime_evidence_v2(
                workspace_root=str(Path(args.workspace_root).resolve()),
                evidence_path=args.artifact_path,
                evidence_sha256=args.expected_sha256,
            )
            payload = _regime_evidence_success_payload(
                status_result,
                status_read=True,
            )
            payload["evidence_path"] = args.artifact_path
            payload["evidence_sha256"] = args.expected_sha256
            _emit(payload)
            return 0
        except RegimeEvidenceV2InputGap as exc:
            code = str(
                getattr(
                    exc,
                    "blocker_code",
                    "TRUE_CURRENT_CANONICAL_INPUT_GAP",
                )
            )
            _emit(
                _regime_evidence_failure_payload(
                    status="TRUE_CURRENT_CANONICAL_INPUT_GAP",
                    blocker_code=code,
                    detail=str(exc),
                )
            )
            return 2
        except RegimeEvidenceV2Error as exc:
            code = str(getattr(exc, "blocker_code", "REGIME_EVIDENCE_V2_BLOCKED"))
            _emit(
                _regime_evidence_failure_payload(
                    status="BLOCKED",
                    blocker_code=code,
                    detail=str(exc),
                )
            )
            return 2
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
