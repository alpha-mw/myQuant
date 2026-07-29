from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from quant_investor.v17_v4_contract import (
    BROKER_AUTHORITY,
    EXECUTION_AUTHORITY,
    FORMAL_RESEARCH_PUBLICATION_AUTHORITY,
    ORDER_AUTHORITY,
    PROTOCOL_VERSION,
    RESEARCH_RUNTIME_DEFAULT,
    TRADE_AUTHORITY,
    canonical_resource_bytes,
    load_canonical_artifact,
    seal_semantic,
    validate_artifact,
    verify_forward_runtime_sources,
    verify_package,
    verify_runtime_build,
)
from quant_investor.v17_v4_contract.resources import (
    PACKAGE_MANIFEST_SHA256,
    load_packaged_json,
)
from quant_investor.v17_v4_contract.schema_validation import (
    SchemaValidationError,
    preflight_schema,
    schema_path_for_version,
    schema_versions,
)
from quant_investor.v17_v4_contract.validators import (
    ArtifactContractError,
    CanaryPointerArtifact,
    CanaryReceiptArtifact,
    CanaryTransitionIntentArtifact,
    DefaultEligibilityIntentArtifact,
    DefaultEligibilityReceiptArtifact,
    DefaultEligiblePointerArtifact,
    DualRunComparisonArtifact,
    FormalActivationIntentArtifact,
    FormalActivationRejectionArtifact,
    FormalActivationReceiptArtifact,
    FormalActivePointerArtifact,
    FormalOutputArtifact,
    HistoricalCanaryPolicyArtifact,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_ROOT = ROOT / "quant_investor" / "v17_v4_contract"
CUTOFF = "2026-07-27T08:00:00Z"
STRATEGY = "quant-first"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _authority(*, formal: bool, default: bool = False) -> dict[str, bool]:
    return {
        "broker": False,
        "execution": False,
        "formal_research_publication": formal,
        "order": False,
        "research_runtime_default": default,
        "trade": False,
    }


def _ref(
    artifact_id: str,
    version: str,
    path: str,
    *,
    byte_sha256: str | None = None,
    cutoff: str = CUTOFF,
) -> dict[str, str]:
    return {
        "artifact_id": artifact_id,
        "artifact_version": version,
        "byte_sha256": byte_sha256 or _sha(f"bytes:{artifact_id}"),
        "cutoff": cutoff,
        "relative_path": path,
        "semantic_sha256": _sha(f"semantic:{artifact_id}"),
        "strategy_id": STRATEGY,
    }


def _ordered_refs(*refs: dict[str, str]) -> list[dict[str, str]]:
    return sorted(
        refs,
        key=lambda row: (
            row["relative_path"],
            row["byte_sha256"],
            row["artifact_id"],
        ),
    )


def _formal_intent() -> dict[str, Any]:
    refs = {
        "formal_output_ref": _ref(
            "formal-output-1",
            "myquant.v17.v4.formal-output.v1",
            "results/v17_v4_formal_research/strategies/quant-first/runs/run-1/formal.json",
        ),
        "source_locator_ref": _ref(
            "source-locator-1",
            "myquant.v17.v4.preselect-locator.v1",
            "data/private/v17_v4_sources/locators/source-1.json",
        ),
        "quant_calibration_receipt_ref": _ref(
            "quant-calibration-1",
            "myquant.v17.v4.calibration-receipt.v1",
            "data/private/v17_v4_runs/run-1/quant-calibration.json",
        ),
        "fundamental_calibration_receipt_ref": _ref(
            "fundamental-calibration-1",
            "myquant.v17.v4.calibration-receipt.v1",
            "data/private/v17_v4_runs/run-1/fundamental-calibration.json",
        ),
        "fusion_promotion_receipt_ref": _ref(
            "fusion-promotion-1",
            "myquant.v17.v4.fusion-promotion-receipt.v1",
            "data/private/v17_v4_runs/run-1/fusion-promotion.json",
        ),
        "deep_bundle_ref": _ref(
            "deep-bundle-1",
            "myquant.v17.v4.deep-evidence-bundle.v1",
            "data/private/v17_v4_runs/run-1/deep-bundle.json",
        ),
        "holdings_snapshot_ref": _ref(
            "holdings-1",
            "myquant.v17.v4.holdings-snapshot.v1",
            "data/private/v17_v4_sources/holdings/holdings-1.json",
        ),
        "risk_policy_ref": _ref(
            "risk-policy-1",
            "myquant.v17.v4.portfolio-risk-policy.v1",
            "data/private/v17_v4_sources/policies/risk-1.json",
        ),
        "macro_overlay_ref": _ref(
            "macro-overlay-1",
            "myquant.v17.v4.portfolio-overlay.v1",
            "data/private/v17_v4_runs/run-1/macro-overlay.json",
        ),
        "markov_overlay_ref": _ref(
            "markov-overlay-1",
            "myquant.v17.v4.portfolio-overlay.v1",
            "data/private/v17_v4_runs/run-1/markov-overlay.json",
        ),
        "factor_control_active_set_ref": _ref(
            "factor-active-set-1",
            ("factor-governance-production-control." "active-set-pointer.schema.v1"),
            "data/private/factor_governance_production_control_v1/active_sets/active.json",
        ),
        "factor_control_activation_receipt_ref": _ref(
            "factor-control-receipt-1",
            ("factor-governance-production-control." "activation-receipt.schema.v1"),
            "data/private/factor_governance_production_control_v1/receipts/control.json",
        ),
        "portfolio_output_ref": _ref(
            "portfolio-output-1",
            "myquant.v17.v4.portfolio-output.v1",
            "data/private/v17_v4_runs/run-1/portfolio-output.json",
        ),
        "package_manifest_ref": _ref(
            "package-manifest",
            "myquant.v17.v4.package-manifest.v1",
            "quant_investor/v17_v4_contract/resources/package_manifest.v1.json",
        ),
        "runtime_manifest_ref": _ref(
            "runtime-manifest",
            "myquant.v17.v4.runtime-build-manifest.v1",
            "quant_investor/v17_v4_contract/resources/runtime_build_manifest.v1.json",
        ),
    }
    return seal_semantic(
        {
            "authority": _authority(formal=False),
            "created_at": CUTOFF,
            "cutoff": CUTOFF,
            **refs,
            "evidence_refs": _ordered_refs(*refs.values()),
            "expected_pointer_sha256": "EMPTY",
            "from_state": "V15_DEFAULT",
            "intent_id": "formal-activation-1",
            "protocol_version": PROTOCOL_VERSION,
            "strategy_id": STRATEGY,
            "version": "myquant.v17.v4.formal-activation-intent.v1",
        }
    )


def _formal_pointer() -> dict[str, Any]:
    intent = _formal_intent()
    return seal_semantic(
        {
            "authority": _authority(formal=False),
            "cutoff": CUTOFF,
            "intent_ref": _ref(
                "formal-activation-1",
                "myquant.v17.v4.formal-activation-intent.v1",
                (
                    "results/v17_v4_formal_research/strategies/"
                    "quant-first/intents/formal-activation-1.json"
                ),
                byte_sha256=hashlib.sha256(canonical_resource_bytes(intent)).hexdigest(),
            ),
            "pointer_id": "formal-pointer-1",
            "protocol_version": PROTOCOL_VERSION,
            "state": "PENDING_COMPLETION",
            "strategy_id": STRATEGY,
            "updated_at": CUTOFF,
            "version": "myquant.v17.v4.formal-active-pointer.v1",
        }
    )


def _formal_receipt() -> dict[str, Any]:
    intent = _formal_intent()
    pointer = _formal_pointer()
    intent_ref = pointer["intent_ref"]
    pointer_ref = _ref(
        "formal-pointer-1",
        "myquant.v17.v4.formal-active-pointer.v1",
        ("results/v17_v4_formal_research/strategies/" "quant-first/_active.json"),
        byte_sha256=hashlib.sha256(canonical_resource_bytes(pointer)).hexdigest(),
    )
    proposed = pointer_ref["byte_sha256"]
    return seal_semantic(
        {
            "authority": _authority(formal=True),
            "cutoff": CUTOFF,
            "evidence_refs": _ordered_refs(intent_ref, pointer_ref),
            "expected_pointer_sha256": intent["expected_pointer_sha256"],
            "from_state": "V15_DEFAULT",
            "intent_ref": intent_ref,
            "observed_pointer_sha256": intent["expected_pointer_sha256"],
            "pointer_ref": pointer_ref,
            "post_readback_sha256": proposed,
            "proposed_pointer_sha256": proposed,
            "protocol_version": PROTOCOL_VERSION,
            "receipt_id": "formal-activation-1",
            "recorded_at": CUTOFF,
            "status": "FORMAL_ACTIVATED",
            "strategy_id": STRATEGY,
            "to_state": "FORMAL_ACTIVE",
            "version": "myquant.v17.v4.formal-activation-receipt.v1",
        }
    )


def _formal_rejection() -> dict[str, Any]:
    return seal_semantic(
        {
            "attempted_evidence_refs": [],
            "authority": _authority(formal=False),
            "expected_pointer_sha256": "EMPTY",
            "from_state": "V15_DEFAULT",
            "observed_pointer_sha256": "EMPTY",
            "protocol_version": PROTOCOL_VERSION,
            "receipt_id": "formal-rejection-1",
            "recorded_at": CUTOFF,
            "rejection_reasons": ["CLOSURE_REVALIDATION_FAILED"],
            "status": "FORMAL_ACTIVATION_REJECTED",
            "strategy_id": STRATEGY,
            "to_state": "V15_DEFAULT",
            "version": "myquant.v17.v4.formal-activation-rejection.v1",
        }
    )


def _formal_output() -> dict[str, Any]:
    return seal_semantic(
        {
            "authority": _authority(formal=True),
            "cutoff": CUTOFF,
            "evidence_refs": [
                _ref(
                    "portfolio-output-1",
                    "myquant.v17.v4.portfolio-output.v1",
                    "data/private/v17_v4_runs/run-1/portfolio-output.json",
                )
            ],
            "output_id": "formal-output-1",
            "protocol_version": PROTOCOL_VERSION,
            "strategy_id": STRATEGY,
            "terminal_state": "PUBLISHED_RESEARCH_ONLY",
            "version": "myquant.v17.v4.formal-output.v1",
        }
    )


def _eligibility_intent() -> dict[str, Any]:
    formal_pointer = _ref(
        "formal-pointer-1",
        "myquant.v17.v4.formal-active-pointer.v1",
        "results/v17_v4_formal_research/strategies/quant-first/_active.json",
    )
    public = [
        _ref(
            f"public-{index}",
            "myquant.v17.v4.public-surface-compatibility-receipt.v1",
            f"data/private/v17_v4_runs/run-1/public-{index}.json",
        )
        for index in range(4)
    ]
    validation = [
        _ref(
            f"validation-{index}",
            "myquant.v17.v4.validation-receipt.v1",
            f"data/private/v17_v4_runs/run-1/validation-{index}.json",
        )
        for index in range(5)
    ]
    bootstrap = _ref(
        "bootstrap-1",
        "myquant.research-runtime.route-bootstrap-receipt.v1",
        "results/research_runtime_control/bootstrap_receipts/bootstrap-1.json",
    )
    rollback = _ref(
        "rollback-drill-1",
        "myquant.v17.v4.rollback-drill-receipt.v1",
        (
            "results/v17_v4_formal_research/strategies/quant-first/"
            "eligibility/rollback_drills/rollback-drill-1.json"
        ),
    )
    return seal_semantic(
        {
            "authority": _authority(formal=False),
            "created_at": CUTOFF,
            "cutoff": CUTOFF,
            "evidence_refs": _ordered_refs(
                formal_pointer,
                bootstrap,
                rollback,
                *public,
                *validation,
            ),
            "expected_pointer_sha256": "EMPTY",
            "formal_active_pointer_ref": formal_pointer,
            "from_state": "FORMAL_ACTIVE",
            "intent_id": "eligibility-1",
            "protocol_version": PROTOCOL_VERSION,
            "public_surface_receipt_refs": _ordered_refs(*public),
            "rollback_drill_receipt_ref": rollback,
            "selector_bootstrap_receipt_ref": bootstrap,
            "strategy_id": STRATEGY,
            "to_state": "DEFAULT_ELIGIBLE",
            "validation_receipt_refs": _ordered_refs(*validation),
            "version": "myquant.v17.v4.default-eligibility-intent.v1",
        }
    )


def _eligible_pointer() -> dict[str, Any]:
    intent = _eligibility_intent()
    return seal_semantic(
        {
            "authority": _authority(formal=False),
            "cutoff": CUTOFF,
            "intent_ref": _ref(
                "eligibility-1",
                "myquant.v17.v4.default-eligibility-intent.v1",
                (
                    "results/v17_v4_formal_research/strategies/quant-first/"
                    "eligibility/intents/eligibility-1.json"
                ),
                byte_sha256=hashlib.sha256(canonical_resource_bytes(intent)).hexdigest(),
            ),
            "pointer_id": "eligible-pointer-1",
            "protocol_version": PROTOCOL_VERSION,
            "state": "PENDING_COMPLETION",
            "strategy_id": STRATEGY,
            "updated_at": CUTOFF,
            "version": "myquant.v17.v4.default-eligible-pointer.v1",
        }
    )


def _eligibility_receipt() -> dict[str, Any]:
    intent = _eligibility_intent()
    pointer = _eligible_pointer()
    intent_ref = pointer["intent_ref"]
    pointer_ref = _ref(
        "eligible-pointer-1",
        "myquant.v17.v4.default-eligible-pointer.v1",
        ("results/v17_v4_formal_research/strategies/quant-first/" "eligibility/_active.json"),
        byte_sha256=hashlib.sha256(canonical_resource_bytes(pointer)).hexdigest(),
    )
    proposed = pointer_ref["byte_sha256"]
    return seal_semantic(
        {
            "authority": _authority(formal=True),
            "cutoff": CUTOFF,
            "evidence_refs": _ordered_refs(intent_ref, pointer_ref),
            "expected_pointer_sha256": intent["expected_pointer_sha256"],
            "from_state": "FORMAL_ACTIVE",
            "intent_ref": intent_ref,
            "observed_pointer_sha256": intent["expected_pointer_sha256"],
            "pointer_ref": pointer_ref,
            "post_readback_sha256": proposed,
            "proposed_pointer_sha256": proposed,
            "protocol_version": PROTOCOL_VERSION,
            "receipt_id": "eligibility-1",
            "recorded_at": CUTOFF,
            "status": "DEFAULT_ELIGIBLE",
            "strategy_id": STRATEGY,
            "to_state": "DEFAULT_ELIGIBLE",
            "version": "myquant.v17.v4.default-eligibility-receipt.v1",
        }
    )


def _canary_intent(*, completed: bool = False) -> dict[str, Any]:
    eligibility = _ref(
        "eligible-pointer-1",
        "myquant.v17.v4.default-eligible-pointer.v1",
        ("results/v17_v4_formal_research/strategies/quant-first/" "eligibility/_active.json"),
    )
    policy = _ref(
        "historical-policy-1",
        "myquant.v17.v4.historical-canary-policy.v1",
        "results/v17_v4_canary/strategies/quant-first/policies/historical-1.json",
    )
    target = _ref(
        "v15-target-1",
        "myquant.research-runtime.protocol-target.v1",
        "results/research_runtime_control/protocol_targets/v15/target.json",
    )
    active = _ref(
        "v15-active-1",
        "myquant.research-runtime.active-run-pointer.v1",
        "results/research_runtime_control/active_runs/v15/quant-first.json",
    )
    comparison_refs = [
        _ref(
            f"operational-comparison-{index}",
            "myquant.v17.v4.dual-run-comparison.v1",
            ("results/v17_v4_canary/strategies/quant-first/" f"runs/{index}/comparison.json"),
        )
        for index in range(5)
    ]
    paired_run_ids = (
        [f"paired-run-{index}" for index in range(1, 6)] if completed else ["paired-run-1"]
    )
    explicit = [eligibility, policy, target, active]
    if completed:
        explicit.extend(comparison_refs)
    completion_fields: dict[str, Any] = {}
    if completed:
        completion_fields = {
            "comparison_refs": comparison_refs,
            "completed_sessions": [
                "2026-07-27",
                "2026-07-28",
                "2026-07-29",
                "2026-07-30",
                "2026-07-31",
            ],
            "side_effect_counters": {
                "active_run_cas_mismatch_count": 0,
                "analysis_time_provider_call_count": 0,
                "broker_call_count": 0,
                "canary_pointer_cas_mismatch_count": 0,
                "data_pointer_cas_mismatch_count": 0,
                "eligibility_pointer_cas_mismatch_count": 0,
                "execution_call_count": 0,
                "factor_pointer_cas_mismatch_count": 0,
                "formal_pointer_cas_mismatch_count": 0,
                "llm_control_call_count": 0,
                "order_call_count": 0,
                "protocol_target_cas_mismatch_count": 0,
                "selector_cas_mismatch_count": 0,
                "trade_call_count": 0,
            },
            "threshold_results": [
                {
                    "observed": "1",
                    "status": "PASS",
                    "threshold_id": "five-of-five",
                }
            ],
        }
    return seal_semantic(
        {
            "authority": _authority(formal=False),
            "created_at": CUTOFF,
            "cutoff": CUTOFF,
            "eligibility_pointer_ref": eligibility,
            "evidence_refs": _ordered_refs(*explicit),
            "expected_pointer_sha256": "EMPTY",
            "from_state": "CANARY" if completed else "DEFAULT_ELIGIBLE",
            "historical_canary_policy_ref": policy,
            "intent_id": ("canary-complete-1" if completed else "canary-start-1"),
            "paired_run_ids": paired_run_ids,
            "protocol_version": PROTOCOL_VERSION,
            "session_window": {
                "end_session": "2026-07-31",
                "required_session_count": 5,
                "start_session": "2026-07-27",
            },
            "strategy_id": STRATEGY,
            "to_state": "CANARY",
            "transition": "COMPLETE" if completed else "START",
            "v15_active_run_pointer_ref": active,
            "v15_protocol_target_ref": target,
            "version": "myquant.v17.v4.canary-transition-intent.v1",
            **completion_fields,
        }
    )


def _canary_pointer(*, completed: bool = False) -> dict[str, Any]:
    intent = _canary_intent(completed=completed)
    intent_id = str(intent["intent_id"])
    return seal_semantic(
        {
            "authority": _authority(formal=False),
            "cutoff": CUTOFF,
            "intent_ref": _ref(
                intent_id,
                "myquant.v17.v4.canary-transition-intent.v1",
                (
                    "results/v17_v4_canary/strategies/quant-first/"
                    f"transitions/intents/{intent_id}.json"
                ),
                byte_sha256=hashlib.sha256(canonical_resource_bytes(intent)).hexdigest(),
            ),
            "pointer_id": f"canary-pointer-{intent_id}",
            "protocol_version": PROTOCOL_VERSION,
            "state": "PENDING_COMPLETION",
            "strategy_id": STRATEGY,
            "updated_at": CUTOFF,
            "version": "myquant.v17.v4.canary-pointer.v1",
        }
    )


def _canary_receipt(*, completed: bool = False) -> dict[str, Any]:
    intent = _canary_intent(completed=completed)
    pointer = _canary_pointer(completed=completed)
    intent_ref = pointer["intent_ref"]
    pointer_ref = _ref(
        str(pointer["pointer_id"]),
        "myquant.v17.v4.canary-pointer.v1",
        "results/v17_v4_canary/strategies/quant-first/_current.json",
        byte_sha256=hashlib.sha256(canonical_resource_bytes(pointer)).hexdigest(),
    )
    proposed = pointer_ref["byte_sha256"]
    return seal_semantic(
        {
            "authority": _authority(formal=True),
            "cutoff": CUTOFF,
            "evidence_refs": _ordered_refs(intent_ref, pointer_ref),
            "expected_pointer_sha256": intent["expected_pointer_sha256"],
            "from_state": intent["from_state"],
            "intent_ref": intent_ref,
            "observed_pointer_sha256": intent["expected_pointer_sha256"],
            "pointer_ref": pointer_ref,
            "post_readback_sha256": proposed,
            "proposed_pointer_sha256": proposed,
            "protocol_version": PROTOCOL_VERSION,
            "receipt_id": str(intent["intent_id"]),
            "recorded_at": CUTOFF,
            "status": "CANARY_COMPLETED" if completed else "CANARY_STARTED",
            "strategy_id": STRATEGY,
            "to_state": intent["to_state"],
            "version": "myquant.v17.v4.canary-receipt.v1",
        }
    )


def _comparison(*, comparable: bool = True, index: int = 0) -> dict[str, Any]:
    def pair(name: str, *, exact: bool = True) -> dict[str, dict[str, str]]:
        shared = _sha(f"{name}:shared")
        return {
            "v15_ref": _ref(
                f"v15-{name}-{index}",
                f"myquant.v15.{name}.v1",
                f"data/private/v15/{name}-{index}.json",
                byte_sha256=shared,
            ),
            "v4_ref": _ref(
                f"v4-{name}-{index}",
                f"myquant.v17.v4.{name}.v1",
                f"data/private/v17_v4_sources/{name}-{index}.json",
                byte_sha256=shared if exact else _sha(f"{name}:different"),
            ),
        }

    inputs = {
        "benchmark": pair("benchmark"),
        "canonical_calendar": pair("canonical-calendar"),
        "holdings_snapshot": pair("holdings-snapshot"),
        "market_bars": pair("market-bars", exact=comparable),
        "source_closure": pair("source-closure", exact=False),
    }
    differing = (
        []
        if comparable
        else _ordered_refs(
            inputs["market_bars"]["v15_ref"],
            inputs["market_bars"]["v4_ref"],
        )
    )
    return seal_semantic(
        {
            "authority": _authority(formal=True),
            "classification": "COMPARABLE" if comparable else "NON_COMPARABLE",
            "comparison_id": f"comparison-{index:02d}",
            "comparison_inputs": inputs,
            "cutoff": CUTOFF,
            "decision_session": "2026-07-27",
            "differing_refs": differing,
            "latency_seconds": {"v15": "30", "v4": "45"},
            "metrics": {
                "cash_exposure_difference": "0.01",
                "cluster_exposure_difference": "0.02",
                "exit_disagreement_count": 0,
                "gross_exposure_difference": "-0.01",
                "held_name_positive_increase_count": 0,
                "industry_exposure_difference": "0.02",
                "l1_portfolio_distance": "0.10",
                "max_common_name_target_difference": "0.01",
                "rank_overlap": "0.75",
                "trim_disagreement_count": 0,
                "turnover_difference": "0.03",
                "v15_top12_recall_in_v4_top24": "0.90",
            },
            "protocol_version": PROTOCOL_VERSION,
            "risk_invariants": {
                "deep_veto_preserved": True,
                "macro_non_increasing": True,
                "markov_non_increasing": True,
                "no_veto_positive_delta": True,
                "permission_veto_preserved": True,
                "v4_gross_excess_within_0_05": True,
            },
            "side_effect_counters": {
                "analysis_time_provider_call_count": 0,
                "broker_call_count": 0,
                "execution_call_count": 0,
                "llm_control_call_count": 0,
                "order_call_count": 0,
                "trade_call_count": 0,
            },
            "stage": "HISTORICAL_REPLAY",
            "strategy_id": STRATEGY,
            "v15_protocol_id": "myquant.v15",
            "v15_run_ref": _ref(
                f"v15-run-{index}",
                "myquant.v15.research-run.v1",
                f"data/private/v15/runs/run-{index}.json",
            ),
            "v4_protocol_id": PROTOCOL_VERSION,
            "v4_run_ref": _ref(
                f"v4-run-{index}",
                "myquant.v17.v4.research-run.v1",
                f"data/private/v17_v4_runs/run-{index}/run.json",
            ),
            "version": "myquant.v17.v4.dual-run-comparison.v1",
        }
    )


def _historical_policy() -> dict[str, Any]:
    refs = [
        _ref(
            f"comparison-{index:02d}",
            "myquant.v17.v4.dual-run-comparison.v1",
            f"results/v17_v4_canary/strategies/quant-first/historical/{index:02d}.json",
            cutoff="2026-07-01T08:00:00Z",
        )
        for index in range(60)
    ]
    return seal_semantic(
        {
            "absolute_risk_limits": {
                "held_name_positive_increase_count": 0,
                "macro_increase_count": 0,
                "markov_increase_count": 0,
                "v4_gross_excess_max": "0.05",
                "veto_positive_delta_count": 0,
            },
            "authority": _authority(formal=True),
            "created_at": CUTOFF,
            "maximum_bands": {
                "cash_exposure_difference": "0.10",
                "cluster_exposure_difference": "0.20",
                "gross_exposure_difference": "0.10",
                "industry_exposure_difference": "0.20",
                "l1_portfolio_distance": "0.30",
                "max_common_name_target_difference": "0.05",
                "turnover_difference": "0.20",
            },
            "minimum_bands": {
                "rank_overlap": "0.50",
                "v15_top12_recall_in_v4_top24": "0.75",
            },
            "origin_count": 60,
            "pair_refs": refs,
            "policy_id": "historical-policy-1",
            "protocol_version": PROTOCOL_VERSION,
            "quantile_method": "empirical_nearest_rank",
            "strategy_id": STRATEGY,
            "version": "myquant.v17.v4.historical-canary-policy.v1",
        }
    )


def test_package_runtime_manifests_and_scaffold_authority_are_sealed() -> None:
    assert PROTOCOL_VERSION == "myquant.v17.v4"
    assert (
        FORMAL_RESEARCH_PUBLICATION_AUTHORITY
        is RESEARCH_RUNTIME_DEFAULT
        is EXECUTION_AUTHORITY
        is BROKER_AUTHORITY
        is ORDER_AUTHORITY
        is TRADE_AUTHORITY
        is False
    )
    verified = verify_package()
    assert verified["resources/package_manifest.v1.json"] == PACKAGE_MANIFEST_SHA256
    assert set(verify_runtime_build()) == {
        "v17_v4_runtime/__init__.py",
        "v17_v4_runtime/authority.py",
        "v17_v4_runtime/calibration.py",
        "v17_v4_runtime/canary_control.py",
        "v17_v4_runtime/cli.py",
        "v17_v4_runtime/deep_control.py",
        "v17_v4_runtime/deep_v2.py",
        "v17_v4_runtime/deep_v3.py",
        "v17_v4_runtime/eligibility_control.py",
        "v17_v4_runtime/formal_activation.py",
        "v17_v4_runtime/factor_observation.py",
        "v17_v4_runtime/forward_evaluation_receipt.py",
        "v17_v4_runtime/forward_evidence.py",
        "v17_v4_runtime/forward_fusion.py",
        "v17_v4_runtime/forward_scoring_v3.py",
        "v17_v4_runtime/forward_shadow.py",
        "v17_v4_runtime/orchestrator.py",
        "v17_v4_runtime/pit_admission.py",
        "v17_v4_runtime/pit_catalog.py",
        "v17_v4_runtime/portfolio_control.py",
        "v17_v4_runtime/public_surfaces.py",
        "v17_v4_runtime/research_factor_set.py",
        "v17_v4_runtime/research_quant.py",
        "v17_v4_runtime/run_profiles.py",
        "v17_v4_runtime/security_directory.py",
        "v17_v4_runtime/shadow_runtime.py",
        "v17_v4_runtime/shadow_prepare_forward.py",
        "v17_v4_runtime/source_storage.py",
        "v17_v4_runtime/tushare_https.py",
    }
    assert set(verify_forward_runtime_sources()) == {
        "factors/forward_evaluator.py",
        "industry/__init__.py",
        "industry/forward_model.py",
        "industry/industry_context.py",
        "industry/industry_evidence_store.py",
        "industry/industry_scorer.py",
        "v17_v4_runtime/themes/__init__.py",
        "v17_v4_runtime/themes/forward_model.py",
    }
    assert not any("research_runtime_control" in path for path in verified)


def test_schema_inventory_is_closed_and_does_not_redefine_neutral_control() -> None:
    assert set(schema_versions()) == {
        "myquant.v17.v4.branch-output.v1",
        "myquant.v17.v4.calibration-origin-inventory.v1",
        "myquant.v17.v4.calibration-receipt.v1",
        "myquant.v17.v4.canary-pointer.v1",
        "myquant.v17.v4.canary-public-snapshot.v1",
        "myquant.v17.v4.canary-receipt.v1",
        "myquant.v17.v4.canary-transition-intent.v1",
        "myquant.v17.v4.deep-evidence-bundle.v1",
        "myquant.v17.v4.deep-evidence-bundle.v2",
        "myquant.v17.v4.deep-evidence-bundle.v3",
        "myquant.v17.v4.deep-assessment-manifest.v1",
        "myquant.v17.v4.deep-assessment-manifest.v2",
        "myquant.v17.v4.default-eligibility-intent.v1",
        "myquant.v17.v4.default-eligibility-receipt.v1",
        "myquant.v17.v4.default-eligible-pointer.v1",
        "myquant.v17.v4.dual-run-comparison.v1",
        "myquant.v17.v4.existing-factor-inventory.v1",
        "myquant.v17.v4.event-scan.v1",
        "myquant.v17.v4.event-scan.v2",
        "myquant.v17.v4.event-scan.v3",
        "myquant.v17.v4.factor-universe-observation.v1",
        "myquant.v17.v4.forward-evaluation-receipt.v1",
        "myquant.v17.v4.forward-evidence-origin-inventory.v1",
        "myquant.v17.v4.forward-factor-allocation.v1",
        "myquant.v17.v4.forward-label.v1",
        "myquant.v17.v4.forward-observation-run.v1",
        "myquant.v17.v4.forward-observation-session-ref.v1",
        "myquant.v17.v4.forward-run-request.v1",
        "myquant.v17.v4.forward-runtime-source-manifest.v1",
        "myquant.v17.v4.forward-stage-output.v1",
        "myquant.v17.v4.forward-stage-receipt.v1",
        "myquant.v17.v4.formal-activation-intent.v1",
        "myquant.v17.v4.formal-activation-receipt.v1",
        "myquant.v17.v4.formal-activation-rejection.v1",
        "myquant.v17.v4.formal-active-pointer.v1",
        "myquant.v17.v4.formal-output.v1",
        "myquant.v17.v4.fusion-promotion-receipt.v1",
        "myquant.v17.v4.fusion-top24.v1",
        "myquant.v17.v4.fusion-top24.v2",
        "myquant.v17.v4.historical-canary-policy.v1",
        "myquant.v17.v4.holdings-snapshot.v1",
        "myquant.v17.v4.initial-pool-output.v1",
        "myquant.v17.v4.issuer-dossier.v1",
        "myquant.v17.v4.issuer-dossier.v2",
        "myquant.v17.v4.issuer-dossier.v3",
        "myquant.v17.v4.official-evidence.v1",
        "myquant.v17.v4.official-evidence.v2",
        "myquant.v17.v4.official-evidence.v3",
        "myquant.v17.v4.pit-catalog-pointer.v1",
        "myquant.v17.v4.pit-generation-catalog.v1",
        "myquant.v17.v4.portfolio-output.v1",
        "myquant.v17.v4.portfolio-overlay.v1",
        "myquant.v17.v4.portfolio-risk-policy.v1",
        "myquant.v17.v4.pretrade-permissions.v1",
        "myquant.v17.v4.preselect-locator.v1",
        "myquant.v17.v4.public-surface-compatibility-receipt.v1",
        "myquant.v17.v4.public-run-dto.v1",
        "myquant.v17.v4.regime-evidence.v1",
        "myquant.v17.v4.research-factor-shadow-assertion.v1",
        "myquant.v17.v4.research-factor-shadow-assertion.v2",
        "myquant.v17.v4.research-factor-input-bundle.v1",
        "myquant.v17.v4.research-fundamental-branch-output.v2",
        "myquant.v17.v4.research-initial-pool-output.v2",
        "myquant.v17.v4.research-quant-branch-output.v1",
        "myquant.v17.v4.research-quant-branch-output.v2",
        "myquant.v17.v4.research-shadow-factor-set-pointer.v1",
        "myquant.v17.v4.research-shadow-factor-set.v1",
        "myquant.v17.v4.research-source-locator.v2",
        "myquant.v17.v4.shadow-fusion-matured-label.v1",
        "myquant.v17.v4.shadow-fusion-observation.v1",
        "myquant.v17.v4.shadow-fusion-policy.v1",
        "myquant.v17.v4.shadow-readiness.v1",
        "myquant.v17.v4.shadow-readiness.v2",
        "myquant.v17.v4.shadow-run.v1",
        "myquant.v17.v4.shadow-run.v2",
        "myquant.v17.v4.shadow-run.v3",
        "myquant.v17.v4.shadow-session-ref.v1",
        "myquant.v17.v4.shadow-session-ref.v2",
        "myquant.v17.v4.shadow-session-ref.v3",
        "myquant.v17.v4.strategy-pool-observation.v1",
        "myquant.v17.v4.rollback-drill-receipt.v1",
        "myquant.v17.v4.total-return-labels.v1",
        "myquant.v17.v4.validation-receipt.v1",
    }
    for path in sorted((CONTRACT_ROOT / "schemas").glob("*.json")):
        schema = load_packaged_json(f"schemas/{path.name}")
        preflight_schema(schema)
        assert not str(schema.get("$id", "")).startswith("myquant.research-runtime.")
    authority = load_packaged_json("schemas/authority.v1.schema.json")["$defs"]["authority"]
    for field in ("broker", "execution", "order", "trade"):
        assert authority["properties"][field]["const"] is False


def test_formal_activation_rejects_research_shadow_run_v2_reference() -> None:
    intent = _formal_intent()
    intent.pop("semantic_sha256")
    intent["formal_output_ref"] = _ref(
        "shadow-run-2",
        "myquant.v17.v4.shadow-run.v2",
        ("results/v17_v4_shadow/strategies/quant-first/" "runs/shadow-run-2.json"),
    )
    with pytest.raises(
        ArtifactContractError,
        match="formal_output_ref artifact version mismatch",
    ):
        validate_artifact(seal_semantic(intent))


def test_formal_transition_and_pointer_separate_publication_from_default() -> None:
    assert isinstance(validate_artifact(_formal_output()), FormalOutputArtifact)
    intent = _formal_intent()
    assert isinstance(
        validate_artifact(intent),
        FormalActivationIntentArtifact,
    )
    assert intent["authority"]["formal_research_publication"] is False
    receipt = _formal_receipt()
    assert isinstance(validate_artifact(receipt), FormalActivationReceiptArtifact)
    assert receipt["authority"]["formal_research_publication"] is True
    assert receipt["authority"]["research_runtime_default"] is False
    pointer = _formal_pointer()
    assert isinstance(validate_artifact(pointer), FormalActivePointerArtifact)

    elevated = dict(receipt)
    elevated["authority"] = _authority(formal=True, default=True)
    elevated.pop("semantic_sha256")
    with pytest.raises(ArtifactContractError, match="authority ceiling"):
        validate_artifact(seal_semantic(elevated))

    execution = dict(receipt)
    execution["authority"] = {**_authority(formal=True), "execution": True}
    execution.pop("semantic_sha256")
    with pytest.raises(SchemaValidationError):
        validate_artifact(seal_semantic(execution))


def test_formal_receipt_rejects_v3_identity_relabel_and_semantic_tamper() -> None:
    intent = _formal_intent()
    v3_ref = dict(intent)
    v3_ref["formal_output_ref"] = {
        **intent["formal_output_ref"],
        "artifact_version": "myquant.v17.v3.formal-research-output.v1",
    }
    v3_ref["evidence_refs"] = _ordered_refs(
        *[
            v3_ref["formal_output_ref"] if row["artifact_id"] == "formal-output-1" else row
            for row in intent["evidence_refs"]
        ]
    )
    v3_ref.pop("semantic_sha256")
    with pytest.raises(ArtifactContractError, match="artifact version mismatch"):
        validate_artifact(seal_semantic(v3_ref))

    receipt = _formal_receipt()
    receipt["status"] = "FORMAL_ACTIVATION_REJECTED"
    with pytest.raises(SchemaValidationError):
        validate_artifact(receipt)

    legacy_pointer = dict(_formal_pointer())
    legacy_pointer["receipt_ref"] = legacy_pointer.pop("intent_ref")
    legacy_pointer["formal_output_ref"] = intent["formal_output_ref"]
    legacy_pointer["state"] = "FORMAL_ACTIVE"
    legacy_pointer.pop("semantic_sha256")
    with pytest.raises(SchemaValidationError):
        validate_artifact(seal_semantic(legacy_pointer))

    v3_artifact = seal_semantic(
        {
            "authority": _authority(formal=False),
            "protocol_version": "myquant.v17.v3",
            "strategy_id": STRATEGY,
            "version": "myquant.v17.v3.activation-receipt.v1",
        }
    )
    with pytest.raises(SchemaValidationError, match="v3 artifact identity"):
        validate_artifact(v3_artifact)


def test_eligibility_and_canary_pointers_never_claim_runtime_default() -> None:
    eligibility_intent = _eligibility_intent()
    assert isinstance(
        validate_artifact(eligibility_intent),
        DefaultEligibilityIntentArtifact,
    )
    eligibility = _eligibility_receipt()
    assert isinstance(
        validate_artifact(eligibility),
        DefaultEligibilityReceiptArtifact,
    )
    assert isinstance(
        validate_artifact(_eligible_pointer()),
        DefaultEligiblePointerArtifact,
    )
    canary_intent = _canary_intent()
    assert isinstance(
        validate_artifact(canary_intent),
        CanaryTransitionIntentArtifact,
    )
    assert isinstance(validate_artifact(_canary_receipt()), CanaryReceiptArtifact)
    assert isinstance(validate_artifact(_canary_pointer()), CanaryPointerArtifact)
    assert eligibility["authority"] == _authority(formal=True)
    assert eligibility_intent["authority"] == _authority(formal=False)
    assert _eligible_pointer()["state"] == "PENDING_COMPLETION"
    assert _canary_pointer()["state"] == "PENDING_COMPLETION"

    completed_intent = _canary_intent(completed=True)
    assert isinstance(
        validate_artifact(completed_intent),
        CanaryTransitionIntentArtifact,
    )
    assert isinstance(
        validate_artifact(_canary_receipt(completed=True)),
        CanaryReceiptArtifact,
    )


def test_dual_run_comparability_is_exact_and_policy_binds_sixty_pairs() -> None:
    comparison = _comparison()
    assert isinstance(validate_artifact(comparison), DualRunComparisonArtifact)
    non_comparable = _comparison(comparable=False, index=1)
    assert isinstance(
        validate_artifact(non_comparable),
        DualRunComparisonArtifact,
    )
    mismatched = _comparison(comparable=False, index=2)
    mismatched["classification"] = "COMPARABLE"
    mismatched.pop("semantic_sha256")
    with pytest.raises(ArtifactContractError, match="exact lower-level bytes"):
        validate_artifact(seal_semantic(mismatched))

    policy = _historical_policy()
    assert isinstance(
        validate_artifact(policy),
        HistoricalCanaryPolicyArtifact,
    )
    policy["pair_refs"].pop()
    policy.pop("semantic_sha256")
    with pytest.raises(SchemaValidationError):
        validate_artifact(seal_semantic(policy))


def test_canonical_loader_rejects_noncanonical_and_additional_properties() -> None:
    receipt = _formal_rejection()
    raw = canonical_resource_bytes(receipt)
    assert isinstance(
        load_canonical_artifact(raw),
        FormalActivationRejectionArtifact,
    )
    with pytest.raises(SchemaValidationError):
        load_canonical_artifact(b" " + raw)
    receipt["neutral_research_runtime_control"] = True
    receipt.pop("semantic_sha256")
    with pytest.raises(SchemaValidationError):
        validate_artifact(seal_semantic(receipt))
    assert schema_path_for_version("myquant.v17.v4.formal-activation-receipt.v1").endswith(
        "formal_activation_receipt.v1.schema.json"
    )
