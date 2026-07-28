from __future__ import annotations

from datetime import date
import hashlib
from pathlib import Path
from typing import Any

import pytest

from quant_investor.v17_v3_contract.canonical import seal_semantic
from quant_investor.v17_v3_contract.resources import (
    PACKAGE_MANIFEST_SHA256,
    load_packaged_json,
)
from quant_investor.v17_v3_contract.schema_validation import (
    SchemaValidationError,
    preflight_schema,
    schema_path_for_version,
    schema_versions,
    validate_artifact,
    validate_instance_against_schema,
)
from quant_investor.v17_v3_contract.validators import (
    ActivationReceiptArtifact,
    ArtifactContractError,
    CalibrationGateInputsArtifact,
    FusionOutputArtifact,
    FusionPromotionReceiptArtifact,
    QuantPreselectionInputsArtifact,
    SourceManifestArtifact,
)

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_ROOT = ROOT / "quant_investor" / "v17_v3_contract" / "schemas"
PROTOCOL = "myquant.v17.v3"
CUTOFF = "2026-07-25T07:00:00Z"
STRATEGY = "quant-first"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _authority(*, formal: bool = False) -> dict[str, bool]:
    return {
        "broker_authority": False,
        "execution_authority": False,
        "formal_research_publication_authority": formal,
        "order_authority": False,
        "production_default": False,
        "trade_authority": False,
    }


def _ref(artifact_id: str, version: str, path: str) -> dict[str, str]:
    return {
        "artifact_id": artifact_id,
        "artifact_version": version,
        "byte_sha256": _sha(f"bytes:{artifact_id}"),
        "cutoff": CUTOFF,
        "relative_path": path,
        "semantic_sha256": _sha(f"semantic:{artifact_id}"),
        "strategy_id": STRATEGY,
    }


def _factor_v4_readiness_ref() -> dict[str, str]:
    return _ref(
        "factor-readiness-1",
        "myquant.v17.v3.factor-governance-readiness.v1",
        "data/private/v17_v3_sources/objects/factor-readiness.json",
    )


def _walk_schema(node: Any, *, path: str = "$") -> None:
    if type(node) is not dict:
        return
    declared = node.get("type")
    types = [declared] if type(declared) is str else declared or []
    if "object" in types:
        assert node.get("additionalProperties") is False, path
    if "array" in types:
        assert type(node.get("uniqueItems")) is bool, path
        assert isinstance(node.get("x-ordering"), str) and node["x-ordering"], path
    for keyword in ("properties", "$defs"):
        for name, child in node.get(keyword, {}).items():
            _walk_schema(child, path=f"{path}.{keyword}.{name}")
    if "items" in node:
        _walk_schema(node["items"], path=f"{path}.items")
    for keyword in ("oneOf", "allOf"):
        for index, child in enumerate(node.get(keyword, [])):
            _walk_schema(child, path=f"{path}.{keyword}[{index}]")


def test_unique_items_supports_full_a_bounds_and_numeric_equality() -> None:
    schema = {
        "$id": "myquant.v17.v3.test-large-unique-array.schema.v1",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "items": {"type": "integer"},
        "maxItems": 10000,
        "type": "array",
        "uniqueItems": True,
        "x-ordering": "integer ascending",
    }
    validate_instance_against_schema(list(range(4097)), schema)
    numeric_schema = {**schema, "items": {"type": "number"}}
    with pytest.raises(SchemaValidationError, match="unique items"):
        validate_instance_against_schema([1, 1.0], numeric_schema)


def _raw_manifest() -> dict[str, Any]:
    roles = [
        "cn_open_day_calendar",
        "corporate_actions",
        "market_bars",
        "pit_fundamentals",
        "universe_membership",
    ]
    return seal_semantic(
        {
            "authority": _authority(),
            "closure_kind": "RAW",
            "created_at": CUTOFF,
            "cutoff": CUTOFF,
            "manifest_id": "raw-1",
            "phase": "RAW",
            "protocol_version": PROTOCOL,
            "sources": [
                {
                    "artifact_ref": _ref(
                        role,
                        f"myquant.v17.v3.{role.replace('_', '-')}.v1",
                        f"data/private/v17_v3_sources/objects/{role}.json",
                    ),
                    "role": role,
                }
                for role in roles
            ],
            "strategy_id": STRATEGY,
            "version": "myquant.v17.v3.source-manifest.v1",
        }
    )


def _month_dates(count: int = 60) -> list[str]:
    year, month = 2018, 1
    values: list[str] = []
    for _ in range(count):
        values.append(date(year, month, 1).isoformat())
        month += 1
        if month == 13:
            year += 1
            month = 1
    return values


def _promotion(*, promoted: bool = True) -> dict[str, Any]:
    origins = _month_dates()
    folds = [
        {
            "fold_index": index + 1,
            "oos_origins": origins[index * 12 : (index + 1) * 12],
            "selected_quant_weight": "0.50",
            "training_origins": origins,
        }
        for index in range(5)
    ]
    common: dict[str, object] = {
        "accepted": promoted,
        "active_refit_origins": origins,
        "authority": _authority(),
        "bootstrap_matrix_sha256": _sha("bootstrap"),
        "calibration_receipt_refs": [
            _ref(
                f"cal-{index}",
                "myquant.v17.v3.fusion-calibration-receipt.v1",
                f"data/private/v17_v3_runs/run-1/cal-{index}.json",
            )
            for index in range(3)
        ],
        "contract_package_manifest_sha256": PACKAGE_MANIFEST_SHA256,
        "created_at": CUTOFF,
        "cutoff": CUTOFF,
        "effective_outer_blocks": 5,
        "evidence_bound": "research_screening_bound",
        "evidence_refs": [
            _ref(
                "fundamental-branch-1",
                "myquant.v17.v3.branch-output.v1",
                "data/private/v17_v3_runs/run-1/fundamental_branch.json",
            ),
            _ref(
                "fusion-calibration-inputs-1",
                "myquant.v17.v3.fusion-calibration-inputs.v1",
                "data/private/v17_v3_runs/run-1/fusion_calibration_inputs.json",
            ),
            _ref(
                "initial-pool-1",
                "myquant.v17.v3.initial-pool-output.v1",
                "data/private/v17_v3_runs/run-1/initial_pool.json",
            ),
            _ref(
                "quant-branch-1",
                "myquant.v17.v3.branch-output.v1",
                "data/private/v17_v3_runs/run-1/quant_branch.json",
            ),
        ],
        "fundamental_branch_policy_sha256": load_packaged_json(
            "resources/fundamental_branch_policy.v1.json"
        )["semantic_sha256"],
        "fold_inventory": folds,
        "fusion_policy_sha256": load_packaged_json("resources/fusion_policy.v1.json")[
            "semantic_sha256"
        ],
        "observation_end_at": CUTOFF,
        "oos_mean_hit60": "0.55",
        "oos_mean_q25_252": "0.10",
        "oos_p5_hit60": "0.51" if promoted else "0.49",
        "oos_p5_q25_252": "0.01",
        "outer_oos_origins": origins,
        "preselector_policy_sha256": load_packaged_json("resources/preselector_policy.v1.json")[
            "semantic_sha256"
        ],
        "promotion_id": "promotion-1",
        "protocol_version": PROTOCOL,
        "quant_branch_policy_sha256": load_packaged_json("resources/quant_branch_policy.v1.json")[
            "semantic_sha256"
        ],
        "status": "PROMOTED" if promoted else "PROMOTION_REJECTED",
        "strategy_id": STRATEGY,
        "version": "myquant.v17.v3.fusion-promotion-receipt.v1",
    }
    if promoted:
        common["active_formal_research_weight"] = "0.50"
    else:
        common["evaluated_quant_weight"] = "0.50"
        common["rejection_reasons"] = ["oos_p5_hit60_not_above_0.50"]
    return seal_semantic(common)


def test_schema_inventory_is_complete_and_every_schema_is_closed() -> None:
    expected = {
        "activation_pointer",
        "activation_receipt",
        "branch_output",
        "calibration_gate_inputs",
        "deep_output",
        "deep_research_inputs",
        "factor_governance_readiness",
        "formal_latest",
        "formal_research_output",
        "fusion_calibration_inputs",
        "fusion_calibration_receipt",
        "fusion_output",
        "fusion_promotion_receipt",
        "initial_pool_output",
        "ledger",
        "portfolio_output",
        "portfolio_overlay",
        "pretrade_permissions",
        "provisional_factor_baseline",
        "quant_preselection_inputs",
        "shadow_latest",
        "shadow_output",
        "source_locator",
        "source_manifest",
        "unpublished_evidence",
    }
    observed = {
        Path(schema_path_for_version(version)).name.removesuffix(".v1.schema.json")
        for version in schema_versions()
    }
    assert observed == expected
    for path in sorted(SCHEMA_ROOT.glob("*.json")):
        schema = load_packaged_json(f"schemas/{path.name}")
        _walk_schema(schema)
        preflight_schema(schema)


def test_raw_manifest_positive_and_derived_cannot_repeat_raw_roles() -> None:
    raw = _raw_manifest()
    assert isinstance(validate_artifact(raw), SourceManifestArtifact)
    derived = seal_semantic(
        {
            "authority": _authority(),
            "closure_kind": "DERIVED_CLOSURE",
            "created_at": CUTOFF,
            "cutoff": CUTOFF,
            "manifest_id": "derived-1",
            "parent_raw_manifest_ref": _ref(
                "raw-1",
                "myquant.v17.v3.source-manifest.v1",
                "data/private/v17_v3_sources/manifests/raw-1.json",
            ),
            "phase": "PRESELECT",
            "protocol_version": PROTOCOL,
            "sources": [
                {
                    "artifact_ref": _ref(
                        "pre-inputs-1",
                        "myquant.v17.v3.quant-preselection-inputs.v1",
                        "data/private/v17_v3_runs/run-1/pre-inputs.json",
                    ),
                    "role": "quant_preselection_inputs",
                }
            ],
            "strategy_id": STRATEGY,
            "version": "myquant.v17.v3.source-manifest.v1",
        }
    )
    assert isinstance(validate_artifact(derived), SourceManifestArtifact)
    derived["sources"].append(raw["sources"][0])
    derived["sources"] = sorted(derived["sources"], key=lambda row: row["role"])
    derived.pop("semantic_sha256")
    with pytest.raises(ArtifactContractError, match="raw or forbidden"):
        validate_artifact(seal_semantic(derived))


def test_promotion_receipt_acceptance_is_metric_auditable() -> None:
    promoted = _promotion(promoted=True)
    result = validate_artifact(promoted)
    assert isinstance(result, FusionPromotionReceiptArtifact)
    assert result.status == "PROMOTED"
    rejected = validate_artifact(_promotion(promoted=False))
    assert isinstance(rejected, FusionPromotionReceiptArtifact)
    assert rejected.status == "PROMOTION_REJECTED"
    promoted["oos_p5_hit60"] = "0.50"
    promoted.pop("semantic_sha256")
    with pytest.raises(ArtifactContractError, match="lower bounds"):
        validate_artifact(seal_semantic(promoted))
    inconsistent = _promotion(promoted=True)
    inconsistent["evidence_refs"][3]["artifact_version"] = "myquant.v17.v3.initial-pool-output.v1"
    inconsistent.pop("semantic_sha256")
    with pytest.raises(ArtifactContractError, match="one initial pool, both branches"):
        validate_artifact(seal_semantic(inconsistent))


def test_calibration_receipt_rejects_empty_evidence_closure() -> None:
    receipt = seal_semantic(
        {
            "version": "myquant.v17.v3.fusion-calibration-receipt.v1",
            "protocol_version": PROTOCOL,
            "calibration_id": "quant-timing-1",
            "calibration_kind": "QUANT_TIMING",
            "strategy_id": STRATEGY,
            "cutoff": CUTOFF,
            "created_at": CUTOFF,
            "observation_end_at": CUTOFF,
            "accepted": True,
            "evidence_refs": [],
            "authority": _authority(),
        }
    )
    with pytest.raises(SchemaValidationError):
        validate_artifact(receipt)


def test_calibration_gate_inputs_bind_kind_role_and_origin_window() -> None:
    document = seal_semantic(
        {
            "version": "myquant.v17.v3.calibration-gate-inputs.v1",
            "protocol_version": PROTOCOL,
            "input_id": "quant-timing-inputs-1",
            "role": "quant_timing_calibration_inputs",
            "calibration_kind": "QUANT_TIMING",
            "strategy_id": STRATEGY,
            "cutoff": CUTOFF,
            "created_at": CUTOFF,
            "observation_start_at": "2021-01-01T07:00:00Z",
            "observation_end_at": CUTOFF,
            "authority": _authority(),
        }
    )
    assert isinstance(
        validate_artifact(document),
        CalibrationGateInputsArtifact,
    )
    document["role"] = "fundamental_forward_calibration_inputs"
    document.pop("semantic_sha256")
    with pytest.raises(ArtifactContractError, match="role/kind"):
        validate_artifact(seal_semantic(document))


def test_preselection_input_binds_exact_packaged_policy() -> None:
    preselector_policy = load_packaged_json("resources/preselector_policy.v1.json")
    quant_policy = load_packaged_json("resources/quant_branch_policy.v1.json")
    preselector_inventory = preselector_policy["factor_inventory"]
    quant_inventory = quant_policy["factor_inventory"]
    document = seal_semantic(
        {
            "authority": _authority(),
            "created_at": CUTOFF,
            "cutoff": CUTOFF,
            "factor_baseline_mode": "FACTOR_V4_PRODUCTION",
            "factor_baseline_ref": _factor_v4_readiness_ref(),
            "input_id": "preselection-inputs-1",
            "payload": {
                "factor_contract": [
                    {
                        "definition_hash": row["definition_sha256"],
                        "family": row["family_id"],
                        "lineage": row["lineage_id"],
                        "lookback": row["lookback_open_days"],
                        "minimum_coverage": "0.95",
                        "name": row["factor_id"],
                        "warmup": row["lookback_open_days"],
                        "weight": row["weight"],
                    }
                    for row in preselector_inventory
                ],
                "observations": [
                    {
                        "data_ready": True,
                        "factor_values": [
                            {"factor_id": row["factor_id"], "value": "1"}
                            for row in preselector_inventory
                        ],
                        "history_count": 120,
                        "liquid": True,
                        "research_eligible": True,
                        "symbol": "000001.SZ",
                        "tradable": True,
                    }
                ],
                "policy_sha256": preselector_policy["semantic_sha256"],
                "quant_branch_inventory": [
                    {
                        "definition_hash": row["definition_sha256"],
                        "family": row["family_id"],
                        "lineage": row["lineage_id"],
                        "name": row["factor_id"],
                    }
                    for row in quant_inventory
                ],
            },
            "protocol_version": PROTOCOL,
            "role": "quant_preselection_inputs",
            "run_id": "run-1",
            "strategy_id": STRATEGY,
            "version": "myquant.v17.v3.quant-preselection-inputs.v1",
        }
    )
    assert isinstance(validate_artifact(document), QuantPreselectionInputsArtifact)
    document["payload"]["policy_sha256"] = _sha("unpackaged-policy")
    document.pop("semantic_sha256")
    with pytest.raises(ArtifactContractError, match="packaged policy"):
        validate_artifact(seal_semantic(document))


def test_ready_fusion_requires_exact_top24() -> None:
    symbols = [f"{index:06d}.SZ" for index in range(1, 25)]
    document = seal_semantic(
        {
            "authority": _authority(),
            "blockers": [],
            "calibration_label": "UNCALIBRATED_50_50",
            "calibration_receipt_refs": [],
            "common_ready_domain": symbols,
            "created_at": CUTOFF,
            "cutoff": CUTOFF,
            "dispositions": [
                {
                    "fundamental_percentile": "0.5",
                    "fusion_score": "0.5",
                    "quant_percentile": "0.5",
                    "reason": None,
                    "selected": True,
                    "status": "READY",
                    "symbol": symbol,
                }
                for symbol in symbols
            ],
            "fundamental_branch_ref": _ref(
                "fundamental-1",
                "myquant.v17.v3.branch-output.v1",
                "data/private/v17_v3_runs/run-1/fundamental.json",
            ),
            "fundamental_weight": "0.50",
            "ordered_domain": symbols,
            "output_id": "fusion-1",
            "promotion_receipt_ref": None,
            "protocol_version": PROTOCOL,
            "quant_branch_ref": _ref(
                "quant-1",
                "myquant.v17.v3.branch-output.v1",
                "data/private/v17_v3_runs/run-1/quant.json",
            ),
            "quant_weight": "0.50",
            "run_id": "run-1",
            "selected_symbols": symbols,
            "state": "FUSION_COMPLETE",
            "status": "READY",
            "strategy_id": STRATEGY,
            "version": "myquant.v17.v3.fusion-output.v1",
        }
    )
    assert isinstance(validate_artifact(document), FusionOutputArtifact)
    document["selected_symbols"] = symbols[:-1]
    document["dispositions"][-1]["selected"] = False
    document.pop("semantic_sha256")
    with pytest.raises(ArtifactContractError, match="incomplete Top24"):
        validate_artifact(seal_semantic(document))


def test_activation_receipt_oneof_requires_promotion_and_status_authority() -> None:
    active = seal_semantic(
        {
            "activated_at": CUTOFF,
            "authority": _authority(formal=True),
            "cutoff": CUTOFF,
            "formal_output_ref": _ref(
                "formal-1",
                "myquant.v17.v3.formal-research-output.v1",
                "results/v17_v3_formal_research/strategies/quant-first/runs/run-1/formal.json",
            ),
            "promotion_receipt_ref": _ref(
                "promotion-1",
                "myquant.v17.v3.fusion-promotion-receipt.v1",
                "data/private/v17_v3_runs/run-1/promotion.json",
            ),
            "protocol_version": PROTOCOL,
            "receipt_id": "active-1",
            "status": "ACTIVE",
            "strategy_id": STRATEGY,
            "version": "myquant.v17.v3.activation-receipt.v1",
        }
    )
    assert isinstance(validate_artifact(active), ActivationReceiptArtifact)
    active.pop("promotion_receipt_ref")
    active.pop("semantic_sha256")
    with pytest.raises(SchemaValidationError):
        validate_artifact(seal_semantic(active))


def test_formal_terminal_mapping_and_portfolio_reference_are_explicit() -> None:
    analyze_ref = _ref(
        "analyze-1",
        "myquant.v17.v3.source-locator.v1",
        "data/private/v17_v3_sources/locators/quant-first/analyze.json",
    )
    factor_baseline_ref = _factor_v4_readiness_ref()
    rank_only = seal_semantic(
        {
            "analyze_locator_ref": analyze_ref,
            "artifact_refs": [analyze_ref, factor_baseline_ref],
            "authority": _authority(),
            "created_at": CUTOFF,
            "cutoff": CUTOFF,
            "factor_baseline_mode": "FACTOR_V4_PRODUCTION",
            "factor_baseline_ref": factor_baseline_ref,
            "output_id": "formal-1",
            "portfolio_basis": None,
            "portfolio_output_ref": None,
            "portfolio_status": "NOT_REQUESTED",
            "protocol_version": PROTOCOL,
            "run_id": "run-1",
            "strategy_id": STRATEGY,
            "terminal_state": "FORMAL_RANK_COMPLETE_NO_PORTFOLIO",
            "version": "myquant.v17.v3.formal-research-output.v1",
        }
    )
    assert validate_artifact(rank_only).payload["portfolio_status"] == "NOT_REQUESTED"
    provisional = dict(rank_only)
    provisional["factor_baseline_mode"] = "PROVISIONAL_RESEARCH"
    provisional.pop("semantic_sha256")
    with pytest.raises(SchemaValidationError):
        validate_artifact(seal_semantic(provisional))
    rank_only["portfolio_status"] = "COMPLETE"
    rank_only.pop("semantic_sha256")
    with pytest.raises(ArtifactContractError, match="terminal state"):
        validate_artifact(seal_semantic(rank_only))


def test_closed_schema_rejects_additional_properties_and_tampering() -> None:
    raw = _raw_manifest()
    raw["unexpected"] = True
    raw.pop("semantic_sha256")
    with pytest.raises(SchemaValidationError):
        validate_artifact(seal_semantic(raw))
    raw = _raw_manifest()
    raw["manifest_id"] = "tampered"
    with pytest.raises(ArtifactContractError, match="semantic_sha256 mismatch"):
        validate_artifact(raw)
