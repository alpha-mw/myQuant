"""Non-operational V17 v5 Phase-0 verification CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any, Sequence

from quant_investor.v17_v5_contract import (
    PROTOCOL_VERSION,
    verify_package,
    verify_predecessor,
    verify_runtime_build,
)
from quant_investor.v17_v5_contract.resources import PackageResourceError
from quant_investor.v17_v5_contract.identities import (
    IdentityContractError,
    require_identifier,
)
from quant_investor.v17_v5_contract.resources import (
    read_packaged_asset,
)
from quant_investor.v17_v5_contract.validators import (
    V4_PACKAGE_MANIFEST_SHA256,
    V4_RUNTIME_MANIFEST_SHA256,
    V4_SOURCE_GIT_COMMIT,
)
from quant_investor.v17_v5_runtime.factor_regime_diagnostics import (
    FactorRegimeDiagnosticError,
    build_unavailable_regime_conditioned_factor_diagnostic,
)
from quant_investor.v17_v5_runtime.v4_compat_reader import (
    V4CompatibilityError,
    read_v4_artifact,
)
from quant_investor.v17_v5_runtime.v4_regime_adapter import (
    V4RegimeAdapterError,
    adapt_v4_regime_evidence,
)

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
    diagnostics = commands.add_parser(
        "factor-regime-diagnostics",
        help="read exact V4 refs and emit one stdout-only descriptive diagnostic",
    )
    diagnostics.add_argument("--workspace-root", required=True)
    diagnostics.add_argument("--strategy-id", required=True)
    diagnostics.add_argument("--factor-name", required=True)
    diagnostics.add_argument("--evaluation-cutoff", required=True)
    diagnostics.add_argument("--created-at", required=True)
    diagnostics.add_argument("--output-id", required=True)
    factor_mode = diagnostics.add_mutually_exclusive_group(required=True)
    factor_mode.add_argument("--factor-evidence-path")
    factor_mode.add_argument("--factor-evidence-unavailable", action="store_true")
    diagnostics.add_argument("--factor-evidence-sha256")
    regime_mode = diagnostics.add_mutually_exclusive_group(required=True)
    regime_mode.add_argument("--regime-evidence-path")
    regime_mode.add_argument("--regime-evidence-unavailable", action="store_true")
    diagnostics.add_argument("--regime-evidence-sha256")
    diagnostics.add_argument("--regime-checkpoint-path")
    diagnostics.add_argument("--regime-checkpoint-sha256")
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


def _factor_regime_policy_ref() -> dict[str, str]:
    policy_path = "quant_investor/v17_v5_contract/resources/factor_regime_diagnostic_policy.v3.json"
    packaged_path = "resources/factor_regime_diagnostic_policy.v3.json"
    raw = read_packaged_asset(packaged_path)
    policy = json.loads(raw)
    return {
        "artifact_id": policy["artifact_id"],
        "byte_sha256": hashlib.sha256(raw).hexdigest(),
        "relative_path": policy_path,
        "semantic_sha256": policy["semantic_sha256"],
        "version": policy["version"],
    }


def _validate_evidence_mode(
    *,
    path: str | None,
    sha256: str | None,
    unavailable: bool,
    label: str,
) -> None:
    if unavailable:
        if sha256 is not None:
            raise V4CompatibilityError(
                f"{label} SHA-256 cannot accompany an unavailable declaration"
            )
        return
    if path is None or sha256 is None:
        raise V4CompatibilityError(f"{label} exact path and SHA-256 are both required")


def _run_factor_regime_diagnostics(args: argparse.Namespace) -> int:
    output_id = require_identifier(args.output_id, label="output_id")
    _validate_evidence_mode(
        path=args.factor_evidence_path,
        sha256=args.factor_evidence_sha256,
        unavailable=args.factor_evidence_unavailable,
        label="factor evidence",
    )
    _validate_evidence_mode(
        path=args.regime_evidence_path,
        sha256=args.regime_evidence_sha256,
        unavailable=args.regime_evidence_unavailable,
        label="regime evidence",
    )
    if args.regime_evidence_unavailable:
        if args.regime_checkpoint_path is not None or args.regime_checkpoint_sha256 is not None:
            raise V4CompatibilityError(
                "regime checkpoint cannot accompany an unavailable regime declaration"
            )
    elif (args.regime_checkpoint_path is None) != (args.regime_checkpoint_sha256 is None):
        raise V4CompatibilityError("regime checkpoint exact path and SHA-256 are both required")
    blockers: list[str] = []
    regime_metadata: dict[str, Any] = {
        "predecessor_package_manifest_sha256": V4_PACKAGE_MANIFEST_SHA256,
        "predecessor_runtime_manifest_sha256": V4_RUNTIME_MANIFEST_SHA256,
        "predecessor_source_commit": V4_SOURCE_GIT_COMMIT,
        "regime_conditioning_eligibility": None,
        "regime_finalized": None,
        "regime_source_version": None,
    }
    if args.factor_evidence_unavailable:
        blockers.append("V4_FACTOR_EVIDENCE_UNAVAILABLE")
    else:
        factor_read = read_v4_artifact(
            args.workspace_root,
            relative_path=args.factor_evidence_path,
            expected_byte_sha256=args.factor_evidence_sha256,
            expected_strategy_id=args.strategy_id,
            decision_cutoff=args.evaluation_cutoff,
        )
        if factor_read.document.get("version") != "myquant.v17.v4.forward-evaluation-receipt.v1":
            raise V4CompatibilityError(
                "factor evidence root must be a V4 forward evaluation receipt"
            )
        lineage_key = factor_read.document.get("lineage_key")
        if (
            factor_read.document.get("subject_id") != args.factor_name
            or type(lineage_key) is not dict
            or lineage_key.get("factor_name") != args.factor_name
        ):
            raise V4CompatibilityError("factor evidence subject does not match --factor-name")
    if args.regime_evidence_unavailable:
        blockers.append("V4_REGIME_EVIDENCE_V3_UNAVAILABLE")
    else:
        regime_read = read_v4_artifact(
            args.workspace_root,
            relative_path=args.regime_evidence_path,
            expected_byte_sha256=args.regime_evidence_sha256,
            expected_strategy_id=args.strategy_id,
            decision_cutoff=args.evaluation_cutoff,
        )
        normalized = adapt_v4_regime_evidence(
            regime_read,
            checkpoint_relative_path=args.regime_checkpoint_path,
            checkpoint_byte_sha256=args.regime_checkpoint_sha256,
        )
        source_version = str(getattr(normalized, "source_version", ""))
        conditioning_eligible = bool(getattr(normalized, "conditioning_eligible", False))
        ineligible_reason = getattr(normalized, "conditioning_ineligibility_reason", None)
        hard_state = getattr(normalized, "hard_state", getattr(normalized, "regime_state", None))
        regime_metadata.update(
            {
                "hard_state": hard_state,
                "continuity_kind": getattr(normalized, "continuity_kind", None),
                "regime_finalized": getattr(normalized, "finalized", False),
                "inference_kind": getattr(normalized, "inference_kind", None),
                "publication_phase": getattr(normalized, "publication_phase", None),
                "regime_conditioning_eligibility": conditioning_eligible,
                "regime_source_version": source_version,
                "scope_kind": getattr(normalized, "scope_kind", None),
                "smoothing_used": getattr(normalized, "smoothing_used", None),
            }
        )
        if source_version != "myquant.v17.v4.regime-evidence.v3":
            blockers.append("V4_REGIME_EVIDENCE_V3_UNAVAILABLE")
            blockers.append(
                str(
                    ineligible_reason
                    or (
                        "REGIME_EVIDENCE_V2_NON_DEPLOYABLE"
                        if source_version == "myquant.v17.v4.regime-evidence.v2"
                        else "REGIME_EVIDENCE_V1_NOT_CONDITIONING_ELIGIBLE"
                    )
                )
            )
        elif not conditioning_eligible:
            blockers.append(str(ineligible_reason or "REGIME_EVIDENCE_NOT_CONDITIONING_ELIGIBLE"))
    origin_binding_result = "NOT_ATTEMPTED"
    if not blockers:
        blockers.append("OBSERVED_FACTOR_REGIME_CLI_PATH_NOT_ENABLED")
        origin_binding_result = "NOT_ENABLED"
    diagnostic = build_unavailable_regime_conditioned_factor_diagnostic(
        strategy_id=args.strategy_id,
        factor_name=args.factor_name,
        factor_implementation_sha256=None,
        policy_ref=_factor_regime_policy_ref(),
        cutoff=args.evaluation_cutoff,
        created_at=args.created_at,
        unavailable_prerequisites=tuple(sorted(set(blockers))),
    )
    wire = _wire()
    _emit(
        {
            **wire,
            "diagnostic": diagnostic,
            "delivery_status": wire["status"],
            "origin_binding_result": origin_binding_result,
            "output_id": output_id,
            "provider_calls": False,
            **regime_metadata,
            "status": diagnostic["status"],
        }
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "status":
        _emit(_wire())
        return 0
    if args.command == "factor-regime-diagnostics":
        try:
            return _run_factor_regime_diagnostics(args)
        except (
            FactorRegimeDiagnosticError,
            IdentityContractError,
            PackageResourceError,
            V4CompatibilityError,
            V4RegimeAdapterError,
        ) as exc:
            _emit(
                {
                    **_wire(),
                    "error": str(exc),
                    "output_id": args.output_id,
                    "provider_calls": False,
                    "verified": False,
                }
            )
            return exc.exit_code
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
