from __future__ import annotations

from datetime import date, timedelta
import hashlib
from typing import Any

import pytest

from quant_investor.v17_v5_contract.canonical import (
    canonical_bytes,
    canonical_resource_bytes,
    seal_semantic,
)
from quant_investor.v17_v5_runtime.v4_compat_reader import (
    V4ClosureNode,
    V4CompatibilityRead,
)
from quant_investor.v17_v5_runtime.v4_factor_adapter import (
    V4FactorAdaptationStatus,
    V4FactorAdapterError,
    adapt_v4_factor_evidence,
    build_factor_diagnostic_from_v4,
)
from quant_investor.v17_v5_contract.validators import (
    V4_COMPATIBILITY_POLICY_BYTE_SHA256,
    V4_PACKAGE_MANIFEST_SHA256,
    V4_RUNTIME_MANIFEST_SHA256,
    V4_SOURCE_GIT_COMMIT,
)

STRATEGY = "quant-first"
FACTOR = "cn_low_total_skewness_20d"
ORIGIN = "2026-01-02"
CUTOFF = "2026-02-02T07:00:00Z"
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _sessions() -> list[str]:
    start = date.fromisoformat(ORIGIN)
    return [(start + timedelta(days=index)).isoformat() for index in range(21)]


def _seal(**values: Any) -> dict[str, Any]:
    return seal_semantic(
        {
            "authority": {
                "broker": False,
                "execution": False,
                "formal_research_publication": False,
                "order": False,
                "research_runtime_default": False,
                "trade": False,
            },
            "protocol_version": "myquant.v17.v4",
            "strategy_id": STRATEGY,
            **values,
        }
    )


def _ref(
    document: dict[str, Any],
    *,
    identity_field: str,
    relative_path: str,
) -> dict[str, str]:
    return {
        "artifact_id": str(document[identity_field]),
        "artifact_version": str(document["version"]),
        "byte_sha256": hashlib.sha256(canonical_resource_bytes(document)).hexdigest(),
        "cutoff": str(document["cutoff"]),
        "relative_path": relative_path,
        "semantic_sha256": str(document["semantic_sha256"]),
        "strategy_id": STRATEGY,
    }


def _read(*, complete: bool = True, corrupt_label_lineage: bool = False) -> V4CompatibilityRead:
    factor_set = _seal(
        audit_session=ORIGIN,
        cutoff=f"{ORIGIN}T06:59:00Z",
        effective_from_session=ORIGIN,
        factor_set_id="factor-set-1",
        selected_factors=[
            {
                "definition_sha256": SHA_A,
                "implementation_resource_sha256": SHA_B,
                "implementation_sha256": SHA_C,
                "name": FACTOR,
            }
        ],
        version="myquant.v17.v4.research-shadow-factor-set.v1",
    )
    factor_set_ref = _ref(
        factor_set,
        identity_field="factor_set_id",
        relative_path=("data/private/v17_v4_sources/research_factor_sets/factor-set-1.json"),
    )
    bundle = _seal(
        bundle_id="bundle-1",
        cutoff=f"{ORIGIN}T07:00:00Z",
        decision_session=ORIGIN,
        factor_set_ref=factor_set_ref,
        factor_slices=[
            {"field_name": field}
            for field in (
                "adj_close",
                "fin_debt_to_assets",
                "fin_ocf_to_profit",
                "fin_roe",
                "total_mv",
            )
        ],
        neutralizer_fields=["beta_252d", "industry", "log_market_cap"],
        required_fields=[
            "adj_close",
            "fin_debt_to_assets",
            "fin_ocf_to_profit",
            "fin_roe",
            "total_mv",
        ],
        source_set_sha256=SHA_B,
        version="myquant.v17.v4.forward-factor-input-bundle.v1",
    )
    bundle_ref = _ref(
        bundle,
        identity_field="bundle_id",
        relative_path=(
            "data/private/v17_v4_sources/snapshots/20260102/" "factor_input_bundle.json"
        ),
    )
    locator = _seal(
        cutoff=f"{ORIGIN}T07:00:00Z",
        decision_session=ORIGIN,
        factor_input_bundle_ref=bundle_ref,
        locator_id="locator-1",
        source_set_sha256=SHA_B,
        version="myquant.v17.v4.forward-source-locator.v1",
    )
    locator_ref = _ref(
        locator,
        identity_field="locator_id",
        relative_path=("data/private/v17_v4_sources/snapshots/20260102/source_locator.json"),
    )
    request = _seal(
        cutoff=f"{ORIGIN}T07:00:00Z",
        decision_session=ORIGIN,
        factor_refs=[factor_set_ref],
        request_id="request-1",
        request_profile="FORWARD_EVIDENCE",
        source_refs=[locator_ref],
        version="myquant.v17.v4.forward-run-request.v1",
    )
    request_ref = _ref(
        request,
        identity_field="request_id",
        relative_path="data/private/v17_v4_runs/forward_requests/request-1.json",
    )
    run = _seal(
        broker=False,
        cutoff=f"{ORIGIN}T07:01:00Z",
        decision_session=ORIGIN,
        execution=False,
        formal_activation_eligible=False,
        global_activation_state="INACTIVE",
        observation_run_id="run-1",
        order=False,
        request_ref=request_ref,
        research_runtime_default=False,
        run_state="FORWARD_EVIDENCE_ACTIVE",
        trade=False,
        version="myquant.v17.v4.forward-observation-run.v1",
    )
    run_ref = _ref(
        run,
        identity_field="observation_run_id",
        relative_path=(
            "results/v17_v4_shadow/forward_evidence/strategies/quant-first/" "runs/run-1.json"
        ),
    )
    observation = _seal(
        completeness="COMPLETE",
        cutoff=f"{ORIGIN}T07:00:30Z",
        decision_session=ORIGIN,
        factor_ref=factor_set_ref,
        observation_id="factor-observation-1",
        observations=[
            {"status": "AVAILABLE", "symbol": "000001.SZ", "value": "1"},
            {"status": "AVAILABLE", "symbol": "600000.SH", "value": "2"},
        ],
        request_ref=request_ref,
        source_refs=[locator_ref],
        version="myquant.v17.v4.factor-universe-observation.v1",
    )
    observation_ref = _ref(
        observation,
        identity_field="observation_id",
        relative_path=(
            "results/v17_v4_shadow/forward_evidence/strategies/quant-first/"
            "observations/factor-observation-1.json"
        ),
    )
    label_lineage = hashlib.sha256(
        canonical_bytes(
            {
                "evidence_refs": [locator_ref],
                "observation_run_ref": run_ref,
                "shanghai_open_sessions": _sessions(),
            }
        )
    ).hexdigest()
    label = _seal(
        completeness="COMPLETE",
        cost_basis_points=20,
        cutoff=CUTOFF,
        decision_session=ORIGIN,
        evidence_refs=[locator_ref],
        horizon_sessions=20,
        label_id="label-1",
        label_rows=[
            {
                "cost_adjusted_return": "0.098",
                "industry_adjusted_return": "0.05",
                "industry_return": "0.05",
                "market_adjusted_return": "0.08",
                "market_return": "0.02",
                "status": "AVAILABLE",
                "symbol": "000001.SZ",
                "total_return": "0.1",
            },
            {
                "cost_adjusted_return": "-0.102",
                "industry_adjusted_return": "-0.05",
                "industry_return": "-0.05",
                "market_adjusted_return": "-0.12",
                "market_return": "0.02",
                "status": "AVAILABLE",
                "symbol": "600000.SH",
                "total_return": "-0.1",
            },
        ],
        label_session=_sessions()[-1],
        observation_run_ref=run_ref,
        shanghai_open_sessions=_sessions(),
        source_lineage_sha256=("0" * 64 if corrupt_label_lineage else label_lineage),
        version="myquant.v17.v4.forward-label.v1",
    )
    label_ref = _ref(
        label,
        identity_field="label_id",
        relative_path=(
            "results/v17_v4_shadow/forward_evidence/strategies/quant-first/" "labels/label-1.json"
        ),
    )
    lineage = {
        "factor_definition_sha256": SHA_A,
        "factor_name": FACTOR,
        "factor_set_sha256": factor_set["semantic_sha256"],
        "horizon_sessions": 20,
        "quant_policy_sha256": SHA_C,
        "source_lineage_sha256": label_lineage,
    }
    origin_inventory = _seal(
        cutoff=CUTOFF,
        decision_session=ORIGIN,
        inventory_id="origin-inventory-1",
        origins=[
            {
                "canonical_evidence_ref": label_ref,
                "duplicate_origin_status": "UNIQUE",
                "evidence_refs": [label_ref],
                "lineage_key": lineage,
                "lineage_key_sha256": hashlib.sha256(canonical_bytes(lineage)).hexdigest(),
                "origin": ORIGIN,
            }
        ],
        request_ref=request_ref,
        version="myquant.v17.v4.forward-evidence-origin-inventory.v1",
    )
    origin_inventory_ref = _ref(
        origin_inventory,
        identity_field="inventory_id",
        relative_path=(
            "results/v17_v4_shadow/forward_evidence/strategies/quant-first/"
            "evaluation/origin-inventory-1.json"
        ),
    )
    factor_inventory = _seal(
        cutoff=CUTOFF,
        decision_session=ORIGIN,
        factors=[
            {
                "definition_sha256": SHA_A,
                "exposure_observation_refs": [observation_ref],
                "factor_name": FACTOR,
                "factor_ref": factor_set_ref,
                "lifecycle": "ACTIVE",
            }
        ],
        inventory_id="factor-inventory-1",
        request_ref=request_ref,
        source_refs=[locator_ref, observation_ref],
        version="myquant.v17.v4.existing-factor-inventory.v1",
    )
    factor_inventory_ref = _ref(
        factor_inventory,
        identity_field="inventory_id",
        relative_path=(
            "results/v17_v4_shadow/forward_evidence/strategies/quant-first/"
            "evaluation/factor-inventory-1.json"
        ),
    )
    receipt = _seal(
        blockers=[] if complete else ["metrics_unavailable"],
        completeness="COMPLETE" if complete else "UNAVAILABLE",
        cutoff=CUTOFF,
        decision_session=ORIGIN,
        evidence_origin_inventory_ref=origin_inventory_ref,
        execution_outcome="SUCCEEDED" if complete else "BLOCKED",
        existing_factor_inventory_ref=factor_inventory_ref,
        label_refs=[label_ref],
        lineage_key=lineage,
        lineage_key_sha256=hashlib.sha256(canonical_bytes(lineage)).hexdigest(),
        observation_run_ref=run_ref,
        origin_count=1,
        receipt_id="receipt-1",
        receipt_type="factor_evaluation_receipt",
        recorded_at=CUTOFF,
        strategy_id=STRATEGY,
        subject_id=FACTOR,
        version="myquant.v17.v4.forward-evaluation-receipt.v1",
    )
    receipt_path = (
        "results/v17_v4_shadow/forward_evidence/strategies/quant-first/" "evaluation/receipt-1.json"
    )
    documents = {
        path: document
        for path, document in (
            (bundle_ref["relative_path"], bundle),
            (factor_inventory_ref["relative_path"], factor_inventory),
            (factor_set_ref["relative_path"], factor_set),
            (label_ref["relative_path"], label),
            (locator_ref["relative_path"], locator),
            (observation_ref["relative_path"], observation),
            (origin_inventory_ref["relative_path"], origin_inventory),
            (receipt_path, receipt),
            (request_ref["relative_path"], request),
            (run_ref["relative_path"], run),
        )
    }
    identity_fields = {
        bundle_ref["relative_path"]: "bundle_id",
        factor_inventory_ref["relative_path"]: "inventory_id",
        factor_set_ref["relative_path"]: "factor_set_id",
        label_ref["relative_path"]: "label_id",
        locator_ref["relative_path"]: "locator_id",
        observation_ref["relative_path"]: "observation_id",
        origin_inventory_ref["relative_path"]: "inventory_id",
        receipt_path: "receipt_id",
        request_ref["relative_path"]: "request_id",
        run_ref["relative_path"]: "observation_run_id",
    }
    closure = tuple(
        V4ClosureNode(
            artifact_id=document[identity_fields[path]],
            byte_sha256=hashlib.sha256(canonical_resource_bytes(document)).hexdigest(),
            relative_path=path,
            semantic_sha256=document["semantic_sha256"],
            validation_mode="V4_REGISTERED_JSON",
            version=document["version"],
        )
        for path, document in sorted(documents.items())
    )
    root_node = next(node for node in closure if node.relative_path == receipt_path)
    return V4CompatibilityRead(
        closure=closure,
        compatibility_policy_byte_sha256=V4_COMPATIBILITY_POLICY_BYTE_SHA256,
        document=receipt,
        documents=documents,
        predecessor_git_commit=V4_SOURCE_GIT_COMMIT,
        predecessor_package_manifest_byte_sha256=V4_PACKAGE_MANIFEST_SHA256,
        predecessor_package_manifest_relative_path=(
            "quant_investor/v17_v4_contract/resources/package_manifest.v1.json"
        ),
        predecessor_protocol_version="myquant.v17.v4",
        predecessor_runtime_manifest_byte_sha256=V4_RUNTIME_MANIFEST_SHA256,
        predecessor_runtime_manifest_relative_path=(
            "quant_investor/v17_v4_contract/resources/runtime_build_manifest.v1.json"
        ),
        root_ref={
            "artifact_id": root_node.artifact_id,
            "artifact_version": root_node.version,
            "byte_sha256": root_node.byte_sha256,
            "cutoff": CUTOFF,
            "relative_path": receipt_path,
            "semantic_sha256": root_node.semantic_sha256,
            "strategy_id": STRATEGY,
        },
        terminal_bindings=(),
    )


def test_adapter_builds_only_accumulating_descriptive_diagnostic() -> None:
    read = _read()

    adaptation = adapt_v4_factor_evidence(
        [read],
        evaluation_cutoff=CUTOFF,
        factor_name=FACTOR,
        open_sessions=_sessions(),
    )
    diagnostic = build_factor_diagnostic_from_v4(
        [read],
        evaluation_cutoff=CUTOFF,
        factor_name=FACTOR,
        open_sessions=_sessions(),
    )

    assert adaptation.status == V4FactorAdaptationStatus.ACCUMULATING
    assert len(adaptation.origins) == 1
    assert len(adaptation.origin_bindings) == 1
    binding = adaptation.origin_bindings[0]
    assert binding.origin_id == adaptation.origins[0].origin_id
    assert binding.factor_implementation_sha256 == SHA_C
    assert binding.eligible_symbol_count == 2
    assert binding.comparable_symbol_count == 2
    assert binding.factor_observation_ref.relative_path.endswith(
        "observations/factor-observation-1.json"
    )
    assert binding.forward_label_ref.relative_path.endswith("labels/label-1.json")
    assert binding.evaluation_receipt_ref.relative_path.endswith("evaluation/receipt-1.json")
    assert binding.observation_run_ref.relative_path.endswith("runs/run-1.json")
    assert binding.request_ref.relative_path.endswith("forward_requests/request-1.json")
    assert binding.source_locator_ref.relative_path.endswith("source_locator.json")
    assert diagnostic["status"] == "ACCUMULATING"
    assert diagnostic["matured_origin_count"] == 1
    assert diagnostic["effectiveness_claimed"] is False
    assert diagnostic["factor_tier_change_eligible"] is False
    assert diagnostic["factor_weight_change_eligible"] is False
    assert diagnostic["promotion_eligible"] is False


def test_adapter_maps_missing_receipts_to_unavailable() -> None:
    adaptation = adapt_v4_factor_evidence(
        [],
        evaluation_cutoff=CUTOFF,
        factor_name=FACTOR,
        open_sessions=_sessions(),
    )
    diagnostic = build_factor_diagnostic_from_v4(
        [],
        evaluation_cutoff=CUTOFF,
        factor_name=FACTOR,
        open_sessions=_sessions(),
    )

    assert adaptation.status == V4FactorAdaptationStatus.UNAVAILABLE
    assert adaptation.origin_bindings == ()
    assert diagnostic["status"] == "UNAVAILABLE"
    assert diagnostic["matured_origin_count"] == 0


def test_adapter_maps_incomplete_receipt_to_unobserved() -> None:
    adaptation = adapt_v4_factor_evidence(
        [_read(complete=False)],
        evaluation_cutoff=CUTOFF,
        factor_name=FACTOR,
        open_sessions=_sessions(),
    )

    assert adaptation.status == V4FactorAdaptationStatus.UNOBSERVED
    assert adaptation.origin_bindings == ()
    assert adaptation.origins == ()
    assert adaptation.stratum is not None


def test_adapter_rejects_source_lineage_tamper_without_artifact() -> None:
    with pytest.raises(V4FactorAdapterError, match="source-lineage mismatch"):
        adapt_v4_factor_evidence(
            [_read(corrupt_label_lineage=True)],
            evaluation_cutoff=CUTOFF,
            factor_name=FACTOR,
            open_sessions=_sessions(),
        )


def test_adapter_deduplicates_identical_origin_and_rejects_mixed_stratum() -> None:
    first = _read()
    adaptation = adapt_v4_factor_evidence(
        [first, first],
        evaluation_cutoff=CUTOFF,
        factor_name=FACTOR,
        open_sessions=_sessions(),
    )
    assert len(adaptation.origins) == 1

    second = _read()
    second.document["lineage_key"]["quant_policy_sha256"] = SHA_A
    with pytest.raises(
        V4FactorAdapterError,
        match="semantic SHA mismatch|lineage SHA mismatch",
    ):
        adapt_v4_factor_evidence(
            [first, second],
            evaluation_cutoff=CUTOFF,
            factor_name=FACTOR,
            open_sessions=_sessions(),
        )
