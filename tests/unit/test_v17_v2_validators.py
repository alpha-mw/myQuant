from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from quant_investor.v17_v2_contract import validators as v17_validators
from quant_investor.v17_v2_contract.canonical import (
    canonical_json_bytes,
    canonical_resource_bytes,
)
from quant_investor.v17_v2_contract.resources import (
    expected_ledger_contract_bindings,
    expected_ledger_implementation_bindings,
)
from quant_investor.v17_v2_contract.validators import (
    ACTION_FAILURE_RECEIPT_VERSION,
    DATASET_MANIFEST_VERSION,
    DATASET_SCHEMA_DIGEST_VERSION,
    DEEP_RESEARCH_REPORT_VERSION,
    DEEP_RESEARCH_REQUEST_VERSION,
    DEEP_RESEARCH_RESPONSE_VERSION,
    GENERATION_CATALOG_VERSION,
    OBSERVATION_DISPOSITION_VERSION,
    SHADOW_LATEST_POINTER_VERSION,
    SHADOW_LEDGER_VERSION,
    SHADOW_OUTPUT_VERSION,
    SOURCE_BINDING_SET_VERSION,
    SOURCE_LOCATOR_VERSION,
    SOURCE_MANIFEST_VERSION,
    SourceAdmissionDisposition,
    V17V2ValidationError,
    admit_runtime_source_hash_dag,
    document_byte_sha256,
    require_runtime_usable_source_role_matrix,
    seal_semantic,
    semantic_sha256,
    validate_action_failure_receipt,
    validate_dataset_manifest,
    validate_deep_research_chain,
    validate_shadow_ledger,
    validate_shadow_ledger_chain,
    validate_shadow_ledger_successor,
    validate_semantic_seal,
    validate_shadow_terminal_chain,
    validate_source_hash_dag,
    validate_source_role_matrix,
)

PACKAGE_ROOT = Path(__file__).parents[2] / "quant_investor" / "v17_v2_contract"
RESOURCE_ROOT = PACKAGE_ROOT / "resources"
CUTOFF = "2026-07-22T00:00:00Z"
ZERO_SHA = "0" * 64
DEEP_COVERAGE = (
    "financial_reports_and_three_statement_reconciliation",
    "normalization_and_reversible_adjustments",
    "segments",
    "management_and_governance",
    "ownership",
    "industry_and_competition",
    "products_and_technology",
    "dcf",
    "reverse_dcf",
    "comparable_companies",
    "sotp_if_applicable",
    "bull_base_bear_scenarios",
    "catalysts",
    "counterevidence",
    "falsification_conditions",
    "continuous_monitoring_items",
)
DEEP_LAYERS = (
    "raw_facts",
    "derived_metrics",
    "research_inferences",
    "investment_judgments",
    "risk_alerts",
)
DEEP_RED_FLAGS = (
    "audit_or_going_concern",
    "restatement_or_three_statement_failure",
    "fraud_or_material_penalty",
    "controller_appropriation_or_pledge_crisis",
    "material_related_party_or_governance_conflict",
    "liquidity_or_refinancing_break",
    "customer_or_supplier_concentration_break",
    "product_or_technology_obsolescence",
    "listing_or_delisting_risk",
    "core_thesis_falsified",
)


def _load_resource(name: str) -> dict[str, Any]:
    payload = json.loads((RESOURCE_ROOT / name).read_bytes())
    assert type(payload) is dict
    return payload


def _ref(document: dict[str, Any], path: str) -> dict[str, Any]:
    id_fields = {
        DATASET_MANIFEST_VERSION: "dataset_id",
        OBSERVATION_DISPOSITION_VERSION: "disposition_id",
        SOURCE_MANIFEST_VERSION: "manifest_id",
        GENERATION_CATALOG_VERSION: "catalog_id",
        SOURCE_BINDING_SET_VERSION: "binding_set_id",
        SOURCE_LOCATOR_VERSION: "locator_id",
        DEEP_RESEARCH_REQUEST_VERSION: "request_id",
        DEEP_RESEARCH_REPORT_VERSION: "report_id",
        SHADOW_LEDGER_VERSION: "run_id",
        SHADOW_OUTPUT_VERSION: "run_id",
    }
    return {
        "artifact_id": document[id_fields[document["version"]]],
        "artifact_version": document["version"],
        "relative_path": path,
        "byte_sha256": document_byte_sha256(document),
        "semantic_sha256": document["semantic_sha256"],
    }


def _sealed(**values: Any) -> dict[str, Any]:
    return seal_semantic(
        {
            "protocol_version": "myquant.v17.v2",
            "authority": False,
            **values,
        }
    )


def _dataset_schema_sha256(schema: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "version": DATASET_SCHEMA_DIGEST_VERSION,
                "schema": schema,
            }
        )
    ).hexdigest()


def _source_dag() -> dict[str, Any]:
    role_matrix = _load_resource("source_role_matrix.v1.json")
    dataset_object = b"x"
    dataset_object_sha = hashlib.sha256(dataset_object).hexdigest()
    dataset_object_path = (
        "data/private/v17_sources/protocol-v2/objects/"
        f"{dataset_object_sha[:2]}/{dataset_object_sha}.blob"
    )
    shard = {
        "logical_name": "market-bars.blob",
        "partition_values": {},
        "object_path": dataset_object_path,
        "byte_sha256": dataset_object_sha,
        "size_bytes": len(dataset_object),
        "row_count": 0,
        "min_key": None,
        "max_key": None,
        "schema_sha256": _dataset_schema_sha256([]),
    }
    content_set_sha = hashlib.sha256(
        canonical_json_bytes(
            {
                "domain": "myquant.v17.v2.dataset-content-set.v1",
                "shards": [shard],
            }
        )
    ).hexdigest()
    dataset_path = "data/private/v17_sources/protocol-v2/manifests/" "market-bars.dataset.json"
    dataset = _sealed(
        version=DATASET_MANIFEST_VERSION,
        dataset_id="market-bars",
        role="market_bars_dataset",
        format="BLOB",
        media_type="application/octet-stream",
        schema=[],
        primary_key=[],
        partition_keys=[],
        sort_keys=[],
        shards=[shard],
        total_row_count=0,
        total_size_bytes=1,
        content_set_sha256=content_set_sha,
    )
    dataset_ref = _ref(dataset, dataset_path)

    disposition_path = (
        "data/private/v17_sources/protocol-v2/manifests/" "market-bars.disposition.json"
    )
    disposition = _sealed(
        version=OBSERVATION_DISPOSITION_VERSION,
        disposition_id="market-bars-disposition",
        scope_id="market-bars",
        stage="RANK",
        market="CN",
        cutoff=CUTOFF,
        dataset_manifest_ref=dataset_ref,
        status="UNREADY",
        effect="MARK_STATISTICALLY_UNREADY",
        reason_code="fixture",
        observation_key_sha256s=[],
    )
    disposition_ref = _ref(disposition, disposition_path)

    source_object = b'{"fixture":true}\n'
    source_object_sha = hashlib.sha256(source_object).hexdigest()
    source_path = (
        "data/private/v17_sources/protocol-v2/objects/"
        f"{source_object_sha[:2]}/{source_object_sha}.json"
    )
    source_ref = {
        "artifact_id": "source-market",
        "artifact_version": "myquant.v17.v2.source-object.v1",
        "relative_path": source_path,
        "byte_sha256": source_object_sha,
        "semantic_sha256": ZERO_SHA,
    }
    generation_source_object = b'{"generation_fixture":true}\n'
    generation_source_sha = hashlib.sha256(generation_source_object).hexdigest()
    generation_source_path = (
        "data/private/v17_sources/protocol-v2/objects/"
        f"{generation_source_sha[:2]}/{generation_source_sha}.json"
    )
    generation_source_ref = {
        "artifact_id": "source-pit-generation",
        "artifact_version": "myquant.v17.v2.source-object.v1",
        "relative_path": generation_source_path,
        "byte_sha256": generation_source_sha,
        "semantic_sha256": ZERO_SHA,
    }
    manifest_path = "data/private/v17_sources/protocol-v2/manifests/source-set.json"
    manifest = _sealed(
        version=SOURCE_MANIFEST_VERSION,
        manifest_id="source-set",
        role_matrix_ref={
            "resource_name": "source_role_matrix.v1.json",
            "resource_version": "myquant.v17.v2.source-role-matrix.v1",
            "byte_sha256": hashlib.sha256(canonical_resource_bytes(role_matrix)).hexdigest(),
        },
        market="CN",
        cutoff=CUTOFF,
        created_at="2026-07-22T00:01:00Z",
        source_ordering="role-source_id-ascending",
        sources=[
            {
                "source_id": "source-market",
                "role": "market_bars_dataset",
                "availability": "AVAILABLE",
                "source_ref": source_ref,
            },
            {
                "source_id": "source-pit-generation",
                "role": "pit_generation_catalog",
                "availability": "AVAILABLE",
                "source_ref": generation_source_ref,
            },
        ],
        dataset_manifest_refs=[dataset_ref],
        observation_disposition_refs=[disposition_ref],
    )
    manifest_ref = _ref(manifest, manifest_path)

    summary = _sealed(
        version="myquant.v17.v2.dataset-summary.v1",
        summary_id="market-summary",
        source_manifest_ref=manifest_ref,
        dataset_manifest_ref=dataset_ref,
        row_count=0,
    )
    summary_sha = document_byte_sha256(summary)
    summary_path = (
        "data/private/v17_sources/protocol-v2/objects/" f"{summary_sha[:2]}/{summary_sha}.json"
    )
    summary_ref = {
        "artifact_id": "market-summary",
        "artifact_version": "myquant.v17.v2.dataset-summary.v1",
        "relative_path": summary_path,
        "byte_sha256": summary_sha,
        "semantic_sha256": summary["semantic_sha256"],
    }
    catalog_path = "data/private/v17_sources/protocol-v2/manifests/market.catalog.json"
    catalog = _sealed(
        version=GENERATION_CATALOG_VERSION,
        catalog_id="market-catalog",
        generation_id="market-generation",
        role="pit_generation_catalog",
        phase="RANK",
        market="CN",
        cutoff=CUTOFF,
        created_at="2026-07-22T00:01:00Z",
        source_manifest_ref=manifest_ref,
        table_ordering=("stage-role-table_id-summary_path-summary_sha-" "dataset_path-dataset_sha"),
        tables=[
            {
                "stage": "RANK",
                "role": "market_bars_dataset",
                "table_id": "market-bars",
                "dataset_manifest_ref": dataset_ref,
                "summary_ref": {
                    **summary_ref,
                    "dataset_manifest_ref": dataset_ref,
                },
                "record_schema_id": "market-bars-v1",
                "primary_key": ["trade_date", "ts_code"],
                "valid_time_field": "trade_date",
                "available_time_field": "available_at",
                "selection_policy": ("available_at_or_before_cutoff_then_latest_valid_revision"),
                "conflict_policy": "conflict_is_invalid_no_fallback",
            }
        ],
    )
    catalog_ref = _ref(catalog, catalog_path)

    binding_path = "data/private/v17_sources/protocol-v2/manifests/source-set.bindings.json"
    binding = {
        "stage": "RANK",
        "role": "market_bars_dataset",
        "catalog_ref": catalog_ref,
        "summary_ref": summary_ref,
        "dataset_manifest_ref": dataset_ref,
        "disposition_id": disposition["disposition_id"],
        "observation_disposition_ref": disposition_ref,
    }
    binding_set = _sealed(
        version=SOURCE_BINDING_SET_VERSION,
        binding_set_id="source-bindings",
        market="CN",
        cutoff=CUTOFF,
        source_manifest_ref=manifest_ref,
        binding_ordering=(
            "stage-role-catalog_path-catalog_sha-summary_path-summary_sha-"
            "dataset_path-dataset_sha-disposition_id"
        ),
        bindings=[binding],
    )
    locator = _sealed(
        version=SOURCE_LOCATOR_VERSION,
        locator_id="source-locator",
        market="CN",
        cutoff=CUTOFF,
        created_at="2026-07-22T00:02:00Z",
        binding_set_ref=_ref(binding_set, binding_path),
    )
    locator_path = "data/private/v17_sources/protocol-v2/locators/source-locator.json"
    return {
        "source_role_matrix": role_matrix,
        "source_objects": {
            dataset_object_path: dataset_object,
            source_path: source_object,
            generation_source_path: generation_source_object,
        },
        "dataset_manifests": {dataset_path: dataset},
        "observation_dispositions": {disposition_path: disposition},
        "source_manifest": manifest,
        "source_manifest_path": manifest_path,
        "generation_catalogs": {catalog_path: catalog},
        "summaries": {summary_path: summary},
        "source_binding_set": binding_set,
        "source_binding_set_path": binding_path,
        "source_locator": locator,
        "source_locator_path": locator_path,
    }


def test_semantic_hash_removes_only_root_seal() -> None:
    nested = {
        "protocol_version": "myquant.v17.v2",
        "version": "myquant.v17.v2.fixture.v1",
        "nested": {"semantic_sha256": "1" * 64},
        "authority": False,
    }
    sealed = seal_semantic(nested)
    assert validate_semantic_seal(sealed) == sealed
    changed = copy.deepcopy(sealed)
    changed["nested"]["semantic_sha256"] = "2" * 64
    assert semantic_sha256(changed) != sealed["semantic_sha256"]
    with pytest.raises(V17V2ValidationError, match="semantic_sha256 mismatch"):
        validate_semantic_seal(changed)


def _parquet_dataset(
    ranges: list[tuple[int | float, int | float]],
) -> tuple[dict[str, Any], dict[str, bytes]]:
    schema = [
        {"name": "trade_date", "logical_type": "int64", "nullable": False},
        {"name": "year", "logical_type": "int64", "nullable": False},
    ]
    schema_sha = _dataset_schema_sha256(schema)
    shards: list[dict[str, Any]] = []
    objects: dict[str, bytes] = {}
    for index, (minimum, maximum) in enumerate(ranges):
        raw = f"parquet-fixture-{index}".encode()
        byte_sha = hashlib.sha256(raw).hexdigest()
        object_path = (
            "data/private/v17_sources/protocol-v2/objects/" f"{byte_sha[:2]}/{byte_sha}.parquet"
        )
        shard = {
            "logical_name": f"year=2026/part-{index:04d}.parquet",
            "partition_values": {"year": 2026},
            "object_path": object_path,
            "byte_sha256": byte_sha,
            "size_bytes": len(raw),
            "row_count": 1,
            "min_key": [minimum],
            "max_key": [maximum],
            "schema_sha256": schema_sha,
        }
        shards.append(shard)
        objects[object_path] = raw
    manifest = _sealed(
        version=DATASET_MANIFEST_VERSION,
        dataset_id="parquet-fixture",
        role="market_bars_dataset",
        format="PARQUET",
        media_type="application/vnd.apache.parquet",
        schema=schema,
        primary_key=["trade_date"],
        partition_keys=["year"],
        sort_keys=["trade_date"],
        shards=shards,
        total_row_count=len(shards),
        total_size_bytes=sum(len(value) for value in objects.values()),
        content_set_sha256=hashlib.sha256(
            canonical_json_bytes(
                {
                    "domain": "myquant.v17.v2.dataset-content-set.v1",
                    "shards": shards,
                }
            )
        ).hexdigest(),
    )
    return manifest, objects


def test_dataset_manifest_accepts_typed_nonoverlapping_total_order() -> None:
    manifest, objects = _parquet_dataset([(1, 10), (11, 20)])
    assert validate_dataset_manifest(manifest, source_objects=objects) == manifest


def test_content_addressed_path_helper_accepts_matching_digest_path() -> None:
    digest = "a" * 64
    v17_validators._require_content_addressed_path(
        f"data/private/v17_sources/protocol-v2/objects/aa/{digest}.parquet",
        byte_sha256=digest,
        label="source object",
    )


def test_dataset_manifest_rejects_unexpected_root_field() -> None:
    manifest, objects = _parquet_dataset([(1, 10), (11, 20)])
    changed = seal_semantic(
        {
            **{key: value for key, value in manifest.items() if key != "semantic_sha256"},
            "unexpected": "field",
        }
    )
    with pytest.raises(V17V2ValidationError, match="additional property"):
        validate_dataset_manifest(changed, source_objects=objects)


def test_dataset_manifest_rejects_overlap_at_equal_boundary() -> None:
    manifest, objects = _parquet_dataset([(1, 10), (10, 20)])
    with pytest.raises(V17V2ValidationError, match="overlapping or duplicate"):
        validate_dataset_manifest(manifest, source_objects=objects)


def test_dataset_manifest_rejects_integral_float_alternate_encoding() -> None:
    manifest, objects = _parquet_dataset([(1.0, 10)])
    with pytest.raises(V17V2ValidationError, match="alternate integer encoding"):
        validate_dataset_manifest(manifest, source_objects=objects)


def test_dataset_manifest_rejects_noncanonical_typed_range_order() -> None:
    manifest, objects = _parquet_dataset([(11, 20), (1, 10)])
    with pytest.raises(V17V2ValidationError, match="canonical complete order"):
        validate_dataset_manifest(manifest, source_objects=objects)


def test_dataset_manifest_rejects_parquet_zero_row_shard() -> None:
    manifest, objects = _parquet_dataset([(1, 10)])
    changed = copy.deepcopy(manifest)
    changed["shards"][0]["row_count"] = 0
    changed["total_row_count"] = 0
    changed["content_set_sha256"] = hashlib.sha256(
        canonical_json_bytes(
            {
                "domain": "myquant.v17.v2.dataset-content-set.v1",
                "shards": changed["shards"],
            }
        )
    ).hexdigest()
    changed = seal_semantic(
        {key: value for key, value in changed.items() if key != "semantic_sha256"}
    )
    with pytest.raises(V17V2ValidationError, match="below minimum"):
        validate_dataset_manifest(changed, source_objects=objects)


def test_dataset_manifest_rejects_blob_nonzero_row_shard() -> None:
    dag = _source_dag()
    manifest = copy.deepcopy(next(iter(dag["dataset_manifests"].values())))
    object_path = manifest["shards"][0]["object_path"]
    manifest["shards"][0]["row_count"] = 1
    manifest["content_set_sha256"] = hashlib.sha256(
        canonical_json_bytes(
            {
                "domain": "myquant.v17.v2.dataset-content-set.v1",
                "shards": manifest["shards"],
            }
        )
    ).hexdigest()
    manifest = seal_semantic(
        {key: value for key, value in manifest.items() if key != "semantic_sha256"}
    )
    with pytest.raises(V17V2ValidationError, match="does not match const"):
        validate_dataset_manifest(
            manifest,
            source_objects={object_path: dag["source_objects"][object_path]},
        )


def test_dataset_manifest_rejects_shard_schema_digest_mismatch() -> None:
    manifest, objects = _parquet_dataset([(1, 10)])
    changed = copy.deepcopy(manifest)
    changed["shards"][0]["schema_sha256"] = "f" * 64
    changed["content_set_sha256"] = hashlib.sha256(
        canonical_json_bytes(
            {
                "domain": "myquant.v17.v2.dataset-content-set.v1",
                "shards": changed["shards"],
            }
        )
    ).hexdigest()
    changed = seal_semantic(
        {key: value for key, value in changed.items() if key != "semantic_sha256"}
    )
    with pytest.raises(V17V2ValidationError, match="schema_sha256 mismatch"):
        validate_dataset_manifest(changed, source_objects=objects)


def test_source_hash_dag_accepts_complete_ordered_closure() -> None:
    dag = _source_dag()
    assert validate_source_hash_dag(**dag) == dag["source_locator"]


def test_source_hash_dag_uses_utc_instants_for_locator_write_last_order() -> None:
    dag = _source_dag()
    locator = copy.deepcopy(dag["source_locator"])
    locator["created_at"] = "2026-07-22T01:30:00+02:00"
    dag["source_locator"] = seal_semantic(
        {key: value for key, value in locator.items() if key != "semantic_sha256"}
    )
    with pytest.raises(V17V2ValidationError, match="locator predates"):
        validate_source_hash_dag(**dag)


@pytest.mark.parametrize(
    "target",
    [
        "catalog_root",
        "catalog_table",
        "binding_set_root",
        "binding_row",
    ],
)
def test_source_hash_dag_rejects_schema_forbidden_extra_fields(target: str) -> None:
    dag = _source_dag()
    if target.startswith("catalog"):
        catalog_path, catalog = next(iter(dag["generation_catalogs"].items()))
        changed_catalog = copy.deepcopy(catalog)
        if target == "catalog_root":
            changed_catalog["unexpected"] = "field"
        else:
            changed_catalog["tables"][0]["unexpected"] = "field"
        changed_catalog = seal_semantic(
            {key: value for key, value in changed_catalog.items() if key != "semantic_sha256"}
        )
        dag["generation_catalogs"] = {catalog_path: changed_catalog}
        binding_set = copy.deepcopy(dag["source_binding_set"])
        binding_set["bindings"][0]["catalog_ref"] = _ref(
            changed_catalog,
            catalog_path,
        )
    else:
        binding_set = copy.deepcopy(dag["source_binding_set"])
        if target == "binding_set_root":
            binding_set["unexpected"] = "field"
        else:
            binding_set["bindings"][0]["unexpected"] = "field"
    binding_set = seal_semantic(
        {key: value for key, value in binding_set.items() if key != "semantic_sha256"}
    )
    dag["source_binding_set"] = binding_set
    locator = copy.deepcopy(dag["source_locator"])
    locator["binding_set_ref"] = _ref(
        binding_set,
        dag["source_binding_set_path"],
    )
    dag["source_locator"] = seal_semantic(
        {key: value for key, value in locator.items() if key != "semantic_sha256"}
    )

    with pytest.raises(V17V2ValidationError, match="additional property"):
        validate_source_hash_dag(**dag)


def test_partial_source_role_matrix_is_structural_but_not_runtime_usable() -> None:
    resource = _load_resource("source_role_matrix.v1.json")
    assert validate_source_role_matrix(resource) == resource
    with pytest.raises(V17V2ValidationError, match="not COMPLETE"):
        require_runtime_usable_source_role_matrix(resource)


def test_source_hash_dag_runtime_admission_is_explicit_and_fail_closed() -> None:
    dag = _source_dag()
    assert validate_source_hash_dag(**dag) == dag["source_locator"]
    with pytest.raises(V17V2ValidationError, match="not COMPLETE"):
        admit_runtime_source_hash_dag(**dag, stored_document_bytes={})


def _complete_runtime_matrix(
    *extra_roles: dict[str, Any],
) -> dict[str, Any]:
    partial = _load_resource("source_role_matrix.v1.json")
    market_role = copy.deepcopy(
        next(row for row in partial["roles"] if row["role"] == "market_bars_dataset")
    )
    catalog_role = copy.deepcopy(
        next(row for row in partial["roles"] if row["role"] == "pit_generation_catalog")
    )
    return {
        **{
            key: copy.deepcopy(value)
            for key, value in partial.items()
            if key not in {"completeness", "runtime_usable", "pending_registry", "roles"}
        },
        "completeness": "COMPLETE",
        "runtime_usable": True,
        "pending_registry": [],
        "roles": sorted(
            [market_role, catalog_role, *copy.deepcopy(extra_roles)],
            key=lambda row: (row["role"].lower(), row["role"]),
        ),
    }


def _runtime_core(
    dag: dict[str, Any],
    matrix: dict[str, Any],
) -> v17_validators.SourceAdmissionOutcome:
    return v17_validators._admit_runtime_source_hash_dag_core(
        source_role_matrix=matrix,
        source_objects=dag["source_objects"],
        dataset_manifests=dag["dataset_manifests"],
        source_manifest=dag["source_manifest"],
        generation_catalogs=dag["generation_catalogs"],
        source_binding_set=dag["source_binding_set"],
        source_locator=dag["source_locator"],
    )


def _unavailable_object_role(
    *,
    role: str = "macro_overlay",
    disposition: str = "SHADOW_RANK_COMPLETE_NO_PORTFOLIO",
) -> dict[str, Any]:
    return {
        "availability_disposition": disposition,
        "kind": "OBJECT",
        "phase": "PORTFOLIO",
        "required": True,
        "role": role,
        "schema_status": "FROZEN",
        "schema_version": "myquant.v17.v2.source-locator.schema.v1",
    }


def _append_source_row(dag: dict[str, Any], row: dict[str, Any]) -> None:
    manifest = copy.deepcopy(dag["source_manifest"])
    manifest["sources"].append(row)
    manifest["sources"].sort(key=lambda item: (item["role"], item["source_id"]))
    dag["source_manifest"] = seal_semantic(
        {key: value for key, value in manifest.items() if key != "semantic_sha256"}
    )


def test_runtime_admission_core_returns_typed_admitted_outcome() -> None:
    dag = _source_dag()
    outcome = _runtime_core(dag, _complete_runtime_matrix())
    assert outcome.disposition is SourceAdmissionDisposition.ADMITTED
    assert outcome.unavailable_required_roles == ()
    assert outcome.locator == dag["source_locator"]


def test_runtime_admission_core_returns_sorted_no_portfolio_roles() -> None:
    dag = _source_dag()
    roles = [
        _unavailable_object_role(role="markov_overlay"),
        _unavailable_object_role(role="macro_overlay"),
    ]
    for role in roles:
        _append_source_row(
            dag,
            {
                "source_id": f"{role['role']}-unavailable",
                "role": role["role"],
                "availability": "UNAVAILABLE",
                "reason": "sealed input unavailable",
            },
        )
    outcome = _runtime_core(dag, _complete_runtime_matrix(*roles))
    assert outcome.disposition is SourceAdmissionDisposition.SHADOW_RANK_COMPLETE_NO_PORTFOLIO
    assert outcome.unavailable_required_roles == ("macro_overlay", "markov_overlay")


def test_runtime_admission_core_reject_disposition_precedes_no_portfolio() -> None:
    dag = _source_dag()
    reject_role = _unavailable_object_role(
        role="rank_required",
        disposition="REJECT_BEFORE_INITIALIZED_ZERO_WRITE",
    )
    no_portfolio_role = _unavailable_object_role(role="macro_overlay")
    for role in (reject_role, no_portfolio_role):
        _append_source_row(
            dag,
            {
                "source_id": f"{role['role']}-unavailable",
                "role": role["role"],
                "availability": "UNAVAILABLE",
                "reason": "sealed input unavailable",
            },
        )
    with pytest.raises(V17V2ValidationError, match="zero-write rejection"):
        _runtime_core(
            dag,
            _complete_runtime_matrix(reject_role, no_portfolio_role),
        )


def test_runtime_admission_core_requires_every_required_availability_row() -> None:
    dag = _source_dag()
    with pytest.raises(V17V2ValidationError, match="lacks an availability row"):
        _runtime_core(
            dag,
            _complete_runtime_matrix(_unavailable_object_role()),
        )


def test_runtime_admission_core_rejects_dataset_role_substitution() -> None:
    dag = _source_dag()
    catalog_path, catalog = next(iter(dag["generation_catalogs"].items()))
    changed = copy.deepcopy(catalog)
    changed["tables"][0]["role"] = "substituted_dataset"
    dag["generation_catalogs"] = {
        catalog_path: seal_semantic(
            {key: value for key, value in changed.items() if key != "semantic_sha256"}
        )
    }
    with pytest.raises(V17V2ValidationError, match="substituted dataset role"):
        _runtime_core(dag, _complete_runtime_matrix())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("kind", "OBJECT", "OBJECT role appears in a DATASET carrier"),
        ("phase", "PORTFOLIO", "role phase mismatch"),
        (
            "schema_version",
            "myquant.v17.v2.source-locator.schema.v1",
            "wrong declared schema",
        ),
    ],
)
def test_runtime_admission_core_rejects_wrong_dataset_declaration(
    field: str,
    value: str,
    message: str,
) -> None:
    dag = _source_dag()
    matrix = _complete_runtime_matrix()
    matrix["roles"][0][field] = value
    with pytest.raises(V17V2ValidationError, match=message):
        _runtime_core(dag, matrix)


def test_runtime_admission_core_rejects_duplicate_dataset_role() -> None:
    dag = _source_dag()
    duplicate = copy.deepcopy(next(iter(dag["dataset_manifests"].values())))
    dag["dataset_manifests"]["duplicate.dataset.json"] = duplicate
    with pytest.raises(V17V2ValidationError, match="closure is not one-to-one"):
        _runtime_core(dag, _complete_runtime_matrix())


def _append_available_locator_object(
    dag: dict[str, Any],
    *,
    raw_transform: Callable[[bytes], bytes] | None = None,
) -> dict[str, Any]:
    locator = dag["source_locator"]
    raw = canonical_resource_bytes(locator)
    if raw_transform is not None:
        raw = raw_transform(raw)
    digest = hashlib.sha256(raw).hexdigest()
    path = "data/private/v17_sources/protocol-v2/objects/" f"{digest[:2]}/{digest}.json"
    dag["source_objects"][path] = raw
    _append_source_row(
        dag,
        {
            "source_id": "market-pointer",
            "role": "market_pointer",
            "availability": "AVAILABLE",
            "source_ref": {
                "artifact_id": locator["locator_id"],
                "artifact_version": SOURCE_LOCATOR_VERSION,
                "relative_path": path,
                "byte_sha256": digest,
                "semantic_sha256": locator["semantic_sha256"],
            },
        },
    )
    return _unavailable_object_role(
        role="market_pointer",
        disposition="REJECT_BEFORE_INITIALIZED_ZERO_WRITE",
    )


def test_runtime_admission_core_rejects_object_without_role_and_phase_carrier() -> None:
    dag = _source_dag()
    object_role = _append_available_locator_object(dag)
    with pytest.raises(V17V2ValidationError, match="does not bind exact role and phase"):
        _runtime_core(dag, _complete_runtime_matrix(object_role))

    wrong_version = copy.deepcopy(dag)
    market_pointer_source = next(
        source
        for source in wrong_version["source_manifest"]["sources"]
        if source["role"] == "market_pointer"
    )
    market_pointer_source["source_ref"]["artifact_version"] = SOURCE_MANIFEST_VERSION
    wrong_version["source_manifest"] = seal_semantic(
        {
            key: value
            for key, value in wrong_version["source_manifest"].items()
            if key != "semantic_sha256"
        }
    )
    with pytest.raises(V17V2ValidationError, match="artifact_version mismatch"):
        _runtime_core(wrong_version, _complete_runtime_matrix(object_role))


def test_runtime_admission_core_rejects_noncanonical_object_bytes() -> None:
    dag = _source_dag()
    object_role = _append_available_locator_object(
        dag,
        raw_transform=lambda raw: b" " + raw,
    )
    with pytest.raises(V17V2ValidationError, match="canonical compact JSON"):
        _runtime_core(dag, _complete_runtime_matrix(object_role))


def test_runtime_admission_core_binds_generation_catalog_role_and_phase() -> None:
    dag = _source_dag()
    outcome = _runtime_core(dag, _complete_runtime_matrix())
    catalog_path, catalog = next(iter(dag["generation_catalogs"].items()))
    assert (
        "pit_generation_catalog",
        catalog["catalog_id"],
        GENERATION_CATALOG_VERSION,
        catalog_path,
        document_byte_sha256(catalog),
        catalog["semantic_sha256"],
    ) in outcome.input_bindings

    wrong_phase = copy.deepcopy(dag)
    wrong_catalog = next(iter(wrong_phase["generation_catalogs"].values()))
    wrong_catalog["phase"] = "PORTFOLIO"
    wrong_phase["generation_catalogs"] = {
        catalog_path: seal_semantic(
            {key: value for key, value in wrong_catalog.items() if key != "semantic_sha256"}
        )
    }
    with pytest.raises(V17V2ValidationError, match="carrier phase mismatch"):
        _runtime_core(wrong_phase, _complete_runtime_matrix())


def test_runtime_admission_core_rejects_detached_or_shared_catalog_carrier() -> None:
    dag = _source_dag()
    catalog_path, catalog = next(iter(dag["generation_catalogs"].items()))
    detached = copy.deepcopy(dag)
    detached["generation_catalogs"] = {}
    with pytest.raises(V17V2ValidationError, match="closure is not one-to-one"):
        _runtime_core(detached, _complete_runtime_matrix())

    duplicate = copy.deepcopy(dag)
    duplicate["generation_catalogs"][f"{catalog_path}.duplicate"] = copy.deepcopy(catalog)
    with pytest.raises(V17V2ValidationError, match="closure is not one-to-one"):
        _runtime_core(duplicate, _complete_runtime_matrix())


def test_runtime_admission_core_rejects_source_object_mapping_substitution() -> None:
    dag = _source_dag()
    json_path = next(
        path for path, raw in dag["source_objects"].items() if raw == b'{"fixture":true}\n'
    )
    dag["source_objects"][json_path] = {"fixture": True}
    assert validate_source_hash_dag(**dag) == dag["source_locator"]
    with pytest.raises(V17V2ValidationError, match="exact stored bytes"):
        _runtime_core(dag, _complete_runtime_matrix())


def test_runtime_role_matrix_rejects_unapproved_complete_rewrite() -> None:
    resource = copy.deepcopy(_load_resource("source_role_matrix.v1.json"))
    resource["completeness"] = "COMPLETE"
    resource["runtime_usable"] = True
    resource["pending_registry"] = []
    for row in resource["roles"]:
        row["schema_status"] = "FROZEN"
        if row["schema_version"] is None:
            row["schema_version"] = "myquant.v17.v2.dataset-manifest.schema.v1"
    with pytest.raises(V17V2ValidationError, match="exact approved frozen resource"):
        require_runtime_usable_source_role_matrix(resource)


def test_runtime_source_role_matrix_requires_no_pending_rows() -> None:
    resource = copy.deepcopy(_load_resource("source_role_matrix.v1.json"))
    resource["completeness"] = "COMPLETE"
    resource["runtime_usable"] = True
    resource["pending_registry"] = []
    with pytest.raises(V17V2ValidationError, match="contains PENDING roles"):
        require_runtime_usable_source_role_matrix(resource)


def test_source_role_matrix_rejects_non_schema_frozen_identity() -> None:
    resource = copy.deepcopy(_load_resource("source_role_matrix.v1.json"))
    frozen = next(row for row in resource["roles"] if row["schema_status"] == "FROZEN")
    frozen["schema_version"] = "myquant.v17.v2.dataset-manifest.v1"
    with pytest.raises(V17V2ValidationError, match="does not match pattern"):
        validate_source_role_matrix(resource)


def test_source_hash_dag_rejects_verification_receipt_role_suffix() -> None:
    dag = _source_dag()
    manifest = copy.deepcopy(dag["source_manifest"])
    manifest["sources"][0]["role"] = "market_bars_dataset_verification_receipt"
    dag["source_manifest"] = seal_semantic(
        {key: value for key, value in manifest.items() if key != "semantic_sha256"}
    )
    with pytest.raises(V17V2ValidationError, match="forbidden role suffix"):
        validate_source_hash_dag(**dag)


def test_source_hash_dag_rejects_duplicate_stage_disposition_identity() -> None:
    dag = _source_dag()
    binding_set = copy.deepcopy(dag["source_binding_set"])
    duplicate = copy.deepcopy(binding_set["bindings"][0])
    binding_set["bindings"].append(duplicate)
    dag["source_binding_set"] = seal_semantic(
        {key: value for key, value in binding_set.items() if key != "semantic_sha256"}
    )
    dag["source_locator"] = seal_semantic(
        {
            **{
                key: value
                for key, value in dag["source_locator"].items()
                if key not in {"semantic_sha256", "binding_set_ref"}
            },
            "binding_set_ref": _ref(
                dag["source_binding_set"],
                dag["source_binding_set_path"],
            ),
        }
    )
    with pytest.raises(V17V2ValidationError, match="duplicate items"):
        validate_source_hash_dag(**dag)


def test_source_hash_dag_rejects_summary_embedded_dataset_drift() -> None:
    dag = _source_dag()
    catalog_path, catalog = next(iter(dag["generation_catalogs"].items()))
    changed_catalog = copy.deepcopy(catalog)
    changed_catalog["tables"][0]["summary_ref"]["dataset_manifest_ref"] = {
        **changed_catalog["tables"][0]["dataset_manifest_ref"],
        "byte_sha256": "f" * 64,
    }
    changed_catalog = seal_semantic(
        {key: value for key, value in changed_catalog.items() if key != "semantic_sha256"}
    )
    dag["generation_catalogs"] = {catalog_path: changed_catalog}
    binding_set = copy.deepcopy(dag["source_binding_set"])
    binding_set["bindings"][0]["catalog_ref"] = _ref(changed_catalog, catalog_path)
    dag["source_binding_set"] = seal_semantic(
        {key: value for key, value in binding_set.items() if key != "semantic_sha256"}
    )
    locator = copy.deepcopy(dag["source_locator"])
    locator["binding_set_ref"] = _ref(
        dag["source_binding_set"],
        dag["source_binding_set_path"],
    )
    dag["source_locator"] = seal_semantic(
        {key: value for key, value in locator.items() if key != "semantic_sha256"}
    )
    with pytest.raises(
        V17V2ValidationError,
        match="catalog summary embedded dataset ref mismatch",
    ):
        validate_source_hash_dag(**dag)


def test_source_hash_dag_rejects_summary_row_count_drift() -> None:
    dag = _source_dag()
    old_summary_path, old_summary = next(iter(dag["summaries"].items()))
    changed_summary = seal_semantic(
        {
            **{
                key: value
                for key, value in old_summary.items()
                if key not in {"semantic_sha256", "row_count"}
            },
            "row_count": 999,
        }
    )
    summary_sha = document_byte_sha256(changed_summary)
    summary_path = (
        "data/private/v17_sources/protocol-v2/objects/" f"{summary_sha[:2]}/{summary_sha}.json"
    )
    summary_ref = {
        "artifact_id": changed_summary["summary_id"],
        "artifact_version": changed_summary["version"],
        "relative_path": summary_path,
        "byte_sha256": document_byte_sha256(changed_summary),
        "semantic_sha256": changed_summary["semantic_sha256"],
        "dataset_manifest_ref": changed_summary["dataset_manifest_ref"],
    }
    dag["summaries"] = {summary_path: changed_summary}

    catalog_path, catalog = next(iter(dag["generation_catalogs"].items()))
    changed_catalog = copy.deepcopy(catalog)
    changed_catalog["tables"][0]["summary_ref"] = summary_ref
    changed_catalog = seal_semantic(
        {key: value for key, value in changed_catalog.items() if key != "semantic_sha256"}
    )
    dag["generation_catalogs"] = {catalog_path: changed_catalog}

    binding_set = copy.deepcopy(dag["source_binding_set"])
    binding_set["bindings"][0]["catalog_ref"] = _ref(changed_catalog, catalog_path)
    binding_set["bindings"][0]["summary_ref"] = {
        key: value for key, value in summary_ref.items() if key != "dataset_manifest_ref"
    }
    dag["source_binding_set"] = seal_semantic(
        {key: value for key, value in binding_set.items() if key != "semantic_sha256"}
    )
    locator = copy.deepcopy(dag["source_locator"])
    locator["binding_set_ref"] = _ref(
        dag["source_binding_set"],
        dag["source_binding_set_path"],
    )
    dag["source_locator"] = seal_semantic(
        {key: value for key, value in locator.items() if key != "semantic_sha256"}
    )

    assert old_summary_path not in dag["summaries"]
    with pytest.raises(V17V2ValidationError, match="summary row_count"):
        validate_source_hash_dag(**dag)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("role", "fundamental_generation_catalog", "lacks AVAILABLE source evidence"),
        ("phase", "PORTFOLIO", "role phase mismatch"),
    ],
)
def test_source_hash_dag_rejects_generation_catalog_carrier_substitution(
    field: str,
    value: str,
    message: str,
) -> None:
    dag = _source_dag()
    catalog_path, catalog = next(iter(dag["generation_catalogs"].items()))
    changed_catalog = copy.deepcopy(catalog)
    changed_catalog[field] = value
    changed_catalog = seal_semantic(
        {key: item for key, item in changed_catalog.items() if key != "semantic_sha256"}
    )
    dag["generation_catalogs"] = {catalog_path: changed_catalog}
    with pytest.raises(V17V2ValidationError, match=message):
        validate_source_hash_dag(**dag)


def test_source_hash_dag_rejects_non_total_binding_order() -> None:
    dag = _source_dag()
    dataset_path, dataset = next(iter(dag["dataset_manifests"].items()))
    alpha_dataset_path = dataset_path.replace("market-bars", "alpha-bars")
    alpha_dataset = seal_semantic(
        {
            **{key: value for key, value in dataset.items() if key != "semantic_sha256"},
            "dataset_id": "alpha-bars",
            "role": "market_bars_dataset",
        }
    )
    alpha_dataset_ref = _ref(alpha_dataset, alpha_dataset_path)
    disposition_path, disposition = next(iter(dag["observation_dispositions"].items()))
    alpha_disposition_path = disposition_path.replace("market-bars", "alpha-bars")
    alpha_disposition = seal_semantic(
        {
            **{
                key: value
                for key, value in disposition.items()
                if key
                not in {
                    "semantic_sha256",
                    "disposition_id",
                    "scope_id",
                    "stage",
                    "dataset_manifest_ref",
                }
            },
            "disposition_id": "alpha-disposition",
            "scope_id": "alpha-bars",
            "stage": "A_STAGE",
            "dataset_manifest_ref": alpha_dataset_ref,
        }
    )
    alpha_disposition_ref = _ref(alpha_disposition, alpha_disposition_path)
    dag["dataset_manifests"][alpha_dataset_path] = alpha_dataset
    dag["observation_dispositions"][alpha_disposition_path] = alpha_disposition

    manifest = copy.deepcopy(dag["source_manifest"])
    manifest["dataset_manifest_refs"].append(alpha_dataset_ref)
    manifest["dataset_manifest_refs"].sort(
        key=lambda item: (
            item["artifact_id"],
            item["relative_path"],
            item["byte_sha256"],
        )
    )
    manifest["observation_disposition_refs"].append(alpha_disposition_ref)
    manifest["observation_disposition_refs"].sort(
        key=lambda item: (
            item["artifact_id"],
            item["relative_path"],
            item["byte_sha256"],
        )
    )
    manifest = seal_semantic(
        {key: value for key, value in manifest.items() if key != "semantic_sha256"}
    )
    dag["source_manifest"] = manifest
    manifest_ref = _ref(manifest, dag["source_manifest_path"])

    _, summary = next(iter(dag["summaries"].items()))
    summary = seal_semantic(
        {
            **{
                key: value
                for key, value in summary.items()
                if key not in {"semantic_sha256", "source_manifest_ref"}
            },
            "source_manifest_ref": manifest_ref,
        }
    )
    summary_sha = document_byte_sha256(summary)
    summary_path = (
        "data/private/v17_sources/protocol-v2/objects/" f"{summary_sha[:2]}/{summary_sha}.json"
    )
    summary_ref = {
        "artifact_id": "market-summary",
        "artifact_version": summary["version"],
        "relative_path": summary_path,
        "byte_sha256": summary_sha,
        "semantic_sha256": summary["semantic_sha256"],
    }
    dag["summaries"] = {summary_path: summary}
    alpha_summary = seal_semantic(
        {
            **{
                key: value
                for key, value in summary.items()
                if key
                not in {
                    "semantic_sha256",
                    "summary_id",
                    "source_manifest_ref",
                    "dataset_manifest_ref",
                }
            },
            "summary_id": "alpha-summary",
            "source_manifest_ref": manifest_ref,
            "dataset_manifest_ref": alpha_dataset_ref,
        }
    )
    alpha_summary_sha = document_byte_sha256(alpha_summary)
    alpha_summary_path = (
        "data/private/v17_sources/protocol-v2/objects/"
        f"{alpha_summary_sha[:2]}/{alpha_summary_sha}.json"
    )
    alpha_summary_ref = {
        "artifact_id": "alpha-summary",
        "artifact_version": alpha_summary["version"],
        "relative_path": alpha_summary_path,
        "byte_sha256": alpha_summary_sha,
        "semantic_sha256": alpha_summary["semantic_sha256"],
    }
    dag["summaries"][alpha_summary_path] = alpha_summary

    catalog_path, catalog = next(iter(dag["generation_catalogs"].items()))
    catalog = copy.deepcopy(catalog)
    catalog["source_manifest_ref"] = manifest_ref
    catalog["tables"][0]["summary_ref"] = {
        **summary_ref,
        "dataset_manifest_ref": catalog["tables"][0]["dataset_manifest_ref"],
    }
    alpha_table = {
        **copy.deepcopy(catalog["tables"][0]),
        "stage": "A_STAGE",
        "role": "market_bars_dataset",
        "table_id": "alpha-bars",
        "dataset_manifest_ref": alpha_dataset_ref,
        "summary_ref": {
            **alpha_summary_ref,
            "dataset_manifest_ref": alpha_dataset_ref,
        },
    }
    catalog["tables"] = [alpha_table, catalog["tables"][0]]
    catalog = seal_semantic(
        {key: value for key, value in catalog.items() if key != "semantic_sha256"}
    )
    dag["generation_catalogs"] = {catalog_path: catalog}
    catalog_ref = _ref(catalog, catalog_path)

    binding_set = copy.deepcopy(dag["source_binding_set"])
    first = binding_set["bindings"][0]
    first["catalog_ref"] = catalog_ref
    first["summary_ref"] = summary_ref
    second = {
        "stage": "A_STAGE",
        "role": "market_bars_dataset",
        "catalog_ref": catalog_ref,
        "summary_ref": alpha_summary_ref,
        "dataset_manifest_ref": alpha_dataset_ref,
        "disposition_id": "alpha-disposition",
        "observation_disposition_ref": alpha_disposition_ref,
    }
    binding_set["source_manifest_ref"] = manifest_ref
    binding_set["bindings"] = [first, second]
    binding_set = seal_semantic(
        {key: value for key, value in binding_set.items() if key != "semantic_sha256"}
    )
    dag["source_binding_set"] = binding_set
    locator = copy.deepcopy(dag["source_locator"])
    locator["binding_set_ref"] = _ref(
        binding_set,
        dag["source_binding_set_path"],
    )
    dag["source_locator"] = seal_semantic(
        {key: value for key, value in locator.items() if key != "semantic_sha256"}
    )

    with pytest.raises(V17V2ValidationError, match="complete total order"):
        validate_source_hash_dag(**dag)


def _deep_chain(evidence_count: int = 1) -> dict[str, Any]:
    evidence_items: list[dict[str, Any]] = []
    for index in range(evidence_count):
        evidence_id = f"evidence-{index:04d}"
        evidence = _sealed(
            version="myquant.v17.v2.sealed-evidence.v1",
            evidence_id=evidence_id,
            fact=f"sealed-{index:04d}",
        )
        evidence_sha = document_byte_sha256(evidence)
        evidence_path = (
            "data/private/v17_sources/protocol-v2/objects/"
            f"{evidence_sha[:2]}/{evidence_sha}.json"
        )
        evidence_items.append(
            {
                "evidence_id": evidence_id,
                "kind": "filing",
                "object_ref": {
                    "artifact_id": evidence_id,
                    "artifact_version": evidence["version"],
                    "relative_path": evidence_path,
                    "byte_sha256": evidence_sha,
                    "semantic_sha256": evidence["semantic_sha256"],
                },
                "layers": list(DEEP_LAYERS),
                "coverage": list(DEEP_COVERAGE),
            }
        )
    first_evidence_id = "evidence-0000"
    first_evidence_ref = evidence_items[0]["object_ref"] if evidence_items else None
    request_path = "results/v17_shadow/protocol-v2/runs/run-1/deep/request.json"
    request = _sealed(
        version=DEEP_RESEARCH_REQUEST_VERSION,
        request_id="request-1",
        run_id="run-1",
        market="CN",
        cutoff=CUTOFF,
        source_locator_ref={
            "artifact_id": "source-locator",
            "artifact_version": SOURCE_LOCATOR_VERSION,
            "relative_path": ("data/private/v17_sources/protocol-v2/locators/source-locator.json"),
            "byte_sha256": "1" * 64,
            "semantic_sha256": "2" * 64,
        },
        deterministic_result_ref={
            "artifact_id": "deterministic-result",
            "artifact_version": "myquant.v17.v2.deterministic-result.v1",
            "relative_path": (
                "results/v17_shadow/protocol-v2/runs/run-1/" "deterministic/result.json"
            ),
            "byte_sha256": "3" * 64,
            "semantic_sha256": "4" * 64,
        },
        template_resource_sha256="5" * 64,
        symbol_ordering="sealed-universe-order",
        symbols=["000001.SZ"],
        evidence_by_symbol=[
            {
                "symbol": "000001.SZ",
                "evidence_ready": bool(evidence_items),
                "evidence": evidence_items,
            }
        ],
    )
    report = _sealed(
        version=DEEP_RESEARCH_REPORT_VERSION,
        report_id="report-1",
        request_ref=_ref(request, request_path),
        run_id="run-1",
        market="CN",
        cutoff=CUTOFF,
        symbol="000001.SZ",
        template_resource_sha256="5" * 64,
        evidence_refs=[first_evidence_ref] if first_evidence_ref is not None else [],
        coverage=[
            {
                "area": area,
                "conclusion": f"covered:{area}",
                "evidence_ids": [first_evidence_id],
            }
            for area in DEEP_COVERAGE
        ],
        layers=[
            {
                "layer": layer,
                "content": f"content:{layer}",
                "evidence_ids": [first_evidence_id],
            }
            for layer in DEEP_LAYERS
        ],
        signals={
            "financial": {"signal": 1, "evidence_ids": [first_evidence_id]},
            "business_model": {"signal": 0.5, "evidence_ids": [first_evidence_id]},
            "industry": {"signal": 0, "evidence_ids": [first_evidence_id]},
            "competitiveness": {"signal": 1, "evidence_ids": [first_evidence_id]},
            "management": {"signal": -0.5, "evidence_ids": [first_evidence_id]},
            "valuation": {"signal": -1, "evidence_ids": [first_evidence_id]},
        },
        severe_red_flags=[
            {"flag": flag, "triggered": False, "evidence_ids": []} for flag in DEEP_RED_FLAGS
        ],
        generated_at="2026-07-22T00:02:00Z",
    )
    report_sha = document_byte_sha256(report)
    report_path = (
        "results/v17_shadow/protocol-v2/models/objects/" f"{report_sha[:2]}/{report_sha}.json"
    )
    response = _sealed(
        version=DEEP_RESEARCH_RESPONSE_VERSION,
        response_id="response-1",
        run_id="run-1",
        market="CN",
        cutoff=CUTOFF,
        request_ref=_ref(request, request_path),
        review_ordering="request-symbol-order",
        review_results=[
            {
                "symbol": "000001.SZ",
                "status": "COMPLETE",
                "research_report_ref": _ref(report, report_path),
            }
        ],
        generated_at="2026-07-22T00:02:00Z",
        received_at="2026-07-22T00:03:00Z",
    )
    return {
        "request": request,
        "request_path": request_path,
        "response": response,
        "reports": {report_path: report},
    }


def test_deep_research_chain_binds_request_template_symbol_and_evidence() -> None:
    chain = _deep_chain()
    assert validate_deep_research_chain(**chain) == chain["response"]


def test_deep_research_chain_rejects_report_evidence_drift() -> None:
    chain = _deep_chain()
    report_path, report = next(iter(chain["reports"].items()))
    changed = copy.deepcopy(report)
    changed["evidence_refs"] = []
    changed = seal_semantic(
        {key: value for key, value in changed.items() if key != "semantic_sha256"}
    )
    changed_sha = document_byte_sha256(changed)
    changed_path = (
        "results/v17_shadow/protocol-v2/models/objects/" f"{changed_sha[:2]}/{changed_sha}.json"
    )
    chain["reports"] = {changed_path: changed}
    response = copy.deepcopy(chain["response"])
    response["review_results"][0]["research_report_ref"] = _ref(changed, changed_path)
    chain["response"] = seal_semantic(
        {key: value for key, value in response.items() if key != "semantic_sha256"}
    )
    with pytest.raises(V17V2ValidationError, match="too few items"):
        validate_deep_research_chain(**chain)


def _install_changed_report(
    chain: dict[str, Any],
    changed: dict[str, Any],
) -> None:
    sealed = seal_semantic(
        {key: value for key, value in changed.items() if key != "semantic_sha256"}
    )
    report_sha = document_byte_sha256(sealed)
    report_path = (
        "results/v17_shadow/protocol-v2/models/objects/" f"{report_sha[:2]}/{report_sha}.json"
    )
    chain["reports"] = {report_path: sealed}
    response = copy.deepcopy(chain["response"])
    response["review_results"][0]["research_report_ref"] = _ref(
        sealed,
        report_path,
    )
    chain["response"] = seal_semantic(
        {key: value for key, value in response.items() if key != "semantic_sha256"}
    )


def test_deep_research_chain_accepts_exact_1024_global_evidence_ids() -> None:
    chain = _deep_chain(1024)
    assert validate_deep_research_chain(**chain) == chain["response"]


def test_deep_research_chain_rejects_1025_global_evidence_ids() -> None:
    chain = _deep_chain(1025)
    with pytest.raises(V17V2ValidationError, match="too many items"):
        validate_deep_research_chain(**chain)


def test_deep_research_chain_rejects_globally_duplicate_evidence_ids() -> None:
    chain = _deep_chain()
    request = copy.deepcopy(chain["request"])
    request["symbols"] = ["000001.SZ", "000002.SZ"]
    duplicate_row = copy.deepcopy(request["evidence_by_symbol"][0])
    duplicate_row["symbol"] = "000002.SZ"
    request["evidence_by_symbol"].append(duplicate_row)
    request = seal_semantic(
        {key: value for key, value in request.items() if key != "semantic_sha256"}
    )
    response = copy.deepcopy(chain["response"])
    response["request_ref"] = _ref(request, chain["request_path"])
    response["review_results"] = [
        {"symbol": symbol, "status": "UNAVAILABLE", "reason": "fixture"}
        for symbol in request["symbols"]
    ]
    response = seal_semantic(
        {key: value for key, value in response.items() if key != "semantic_sha256"}
    )
    with pytest.raises(V17V2ValidationError, match="duplicate request evidence_id"):
        validate_deep_research_chain(
            request=request,
            request_path=chain["request_path"],
            response=response,
            reports={},
        )


def test_deep_research_chain_rejects_tampered_signal_evidence_id() -> None:
    chain = _deep_chain()
    report = copy.deepcopy(next(iter(chain["reports"].values())))
    report["signals"]["financial"]["evidence_ids"] = ["outside-request"]
    _install_changed_report(chain, report)
    with pytest.raises(V17V2ValidationError, match="outside the sealed request"):
        validate_deep_research_chain(**chain)


def test_deep_research_chain_rejects_extra_report_fields() -> None:
    chain = _deep_chain()
    report = copy.deepcopy(next(iter(chain["reports"].values())))
    report["unsealed_extension"] = "forbidden"
    _install_changed_report(chain, report)
    with pytest.raises(V17V2ValidationError, match="additional property"):
        validate_deep_research_chain(**chain)


def test_deep_research_chain_requires_evidence_for_triggered_red_flag() -> None:
    chain = _deep_chain()
    report = copy.deepcopy(next(iter(chain["reports"].values())))
    report["severe_red_flags"][0]["triggered"] = True
    _install_changed_report(chain, report)
    with pytest.raises(V17V2ValidationError, match="must be nonempty"):
        validate_deep_research_chain(**chain)


def test_deep_research_chain_rejects_fixed_coverage_order_tamper() -> None:
    chain = _deep_chain()
    report = copy.deepcopy(next(iter(chain["reports"].values())))
    report["coverage"][0], report["coverage"][1] = (
        report["coverage"][1],
        report["coverage"][0],
    )
    _install_changed_report(chain, report)
    with pytest.raises(V17V2ValidationError, match="coverage order or completeness"):
        validate_deep_research_chain(**chain)


def _terminal_chain() -> dict[str, Any]:
    source_dag = _source_dag()
    unavailable_role = _unavailable_object_role(role="macro_overlay")
    _append_source_row(
        source_dag,
        {
            "source_id": "macro-overlay-unavailable",
            "role": "macro_overlay",
            "availability": "UNAVAILABLE",
            "reason": "sealed input unavailable",
        },
    )
    source_admission = _runtime_core(
        source_dag,
        _complete_runtime_matrix(unavailable_role),
    )
    ledger_path = "results/v17_shadow/protocol-v2/runs/run-1/ledger.json"
    states = [
        "PREPARED",
        "DETERMINISTIC_COMPLETE",
        "DEEP_REQUEST_READY",
        "DEEP_RESPONSE_RECEIVED",
        "SHADOW_RANK_COMPLETE_NO_PORTFOLIO",
    ]
    actions = [
        "SHADOW_PREPARE",
        "SHADOW_PREPARE",
        "SHADOW_PREPARE",
        "SHADOW_RECEIVE",
        "SHADOW_FINALIZE",
    ]
    times = [
        "2026-07-22T00:03:00Z",
        "2026-07-22T00:04:00Z",
        "2026-07-22T00:05:00Z",
        "2026-07-22T00:06:00Z",
        "2026-07-22T00:07:00Z",
    ]
    locator = source_admission.locator
    locator_ref = {
        "artifact_id": locator["locator_id"],
        "artifact_version": SOURCE_LOCATOR_VERSION,
        "relative_path": (
            "data/private/v17_sources/protocol-v2/locators/" f"{locator['locator_id']}.json"
        ),
        "byte_sha256": source_admission.locator_byte_sha256,
        "semantic_sha256": locator["semantic_sha256"],
    }
    locator_binding = {
        "locator_id": locator["locator_id"],
        "locator_ref": locator_ref,
    }
    input_bindings: list[dict[str, Any]] = [
        {
            "role": role,
            "artifact_ref": {
                "artifact_id": artifact_id,
                "artifact_version": artifact_version,
                "relative_path": relative_path,
                "byte_sha256": byte_sha256,
                "semantic_sha256": semantic_sha256_value,
            },
        }
        for (
            role,
            artifact_id,
            artifact_version,
            relative_path,
            byte_sha256,
            semantic_sha256_value,
        ) in source_admission.input_bindings
    ]
    input_binding_sha256s = sorted(
        binding["artifact_ref"]["byte_sha256"] for binding in input_bindings
    )
    history: list[dict[str, Any]] = []
    ledger_bytes: list[bytes] = []
    ledger: dict[str, Any] = {}
    for sequence, (state, action, at) in enumerate(zip(states, actions, times, strict=True)):
        predecessor_sha = "EMPTY" if sequence == 0 else hashlib.sha256(ledger_bytes[-1]).hexdigest()
        history.append(
            {
                "sequence": sequence,
                "attempt_id": f"attempt-{sequence}",
                "action": action,
                "acceptance_checkpoint": "INITIALIZED",
                "from_state": None if sequence == 0 else states[sequence - 1],
                "to_state": state,
                "at": at,
                "expected_ledger_sha256": predecessor_sha,
                "input_binding_sha256s": input_binding_sha256s,
                "artifact_roles": [],
            }
        )
        ledger = _sealed(
            version=SHADOW_LEDGER_VERSION,
            run_id="run-1",
            strategy_id="cn-shadow",
            market="CN",
            cutoff=CUTOFF,
            state=state,
            sequence=sequence,
            action=action,
            checkpoint="INITIALIZED",
            created_at=times[0],
            updated_at=at,
            previous_ledger_sha256=predecessor_sha,
            locator_binding=locator_binding,
            contract_bindings=expected_ledger_contract_bindings(),
            implementation_bindings=expected_ledger_implementation_bindings(),
            input_bindings=input_bindings,
            artifacts=[],
            history=copy.deepcopy(history),
        )
        ledger_bytes.append(canonical_resource_bytes(ledger))
    output_path = "results/v17_shadow/protocol-v2/outcomes/run-1.json"
    output = _sealed(
        version=SHADOW_OUTPUT_VERSION,
        run_id="run-1",
        strategy_id="cn-shadow",
        market="CN",
        cutoff=CUTOFF,
        terminal_state="SHADOW_RANK_COMPLETE_NO_PORTFOLIO",
        ledger_ref=_ref(ledger, ledger_path),
        source_locator_ref=ledger["locator_binding"]["locator_ref"],
        rank_output={},
        portfolio_output=None,
        blockers=[],
        generated_at="2026-07-22T00:07:00Z",
    )
    latest = _sealed(
        version=SHADOW_LATEST_POINTER_VERSION,
        pointer_path="results/v17_shadow/protocol-v2/_latest/shadow.json",
        run_id="run-1",
        terminal_state="SHADOW_RANK_COMPLETE_NO_PORTFOLIO",
        ledger_ref=_ref(ledger, ledger_path),
        terminal_output_ref=_ref(output, output_path),
        previous_pointer_byte_sha256="EMPTY",
        publication_mode="NORMAL",
        published_at="2026-07-22T00:08:00Z",
    )
    return {
        "ledger": ledger,
        "ledger_bytes": ledger_bytes[-1],
        "predecessor_ledger_bytes": ledger_bytes[:-1],
        "ledger_path": ledger_path,
        "output": output,
        "output_bytes": canonical_resource_bytes(output),
        "output_path": output_path,
        "latest_pointer": latest,
        "previous_pointer_bytes": None,
        "source_admission": source_admission,
    }


def _validate_admitted_terminal_chain(chain: dict[str, Any]) -> dict[str, Any]:
    return v17_validators._validate_shadow_terminal_chain_admitted(
        ledger_bytes=chain["ledger_bytes"],
        predecessor_ledger_bytes=chain["predecessor_ledger_bytes"],
        ledger_path=chain["ledger_path"],
        output_bytes=chain["output_bytes"],
        output_path=chain["output_path"],
        latest_pointer=chain["latest_pointer"],
        previous_pointer_bytes=chain["previous_pointer_bytes"],
        source_admission=chain["source_admission"],
    )


def _install_terminal_ledger(chain: dict[str, Any], ledger: dict[str, Any]) -> None:
    sealed_ledger = seal_semantic(
        {key: value for key, value in ledger.items() if key != "semantic_sha256"}
    )
    chain["ledger"] = sealed_ledger
    chain["ledger_bytes"] = canonical_resource_bytes(sealed_ledger)
    output = copy.deepcopy(chain["output"])
    output["ledger_ref"] = _ref(sealed_ledger, chain["ledger_path"])
    sealed_output = seal_semantic(
        {key: value for key, value in output.items() if key != "semantic_sha256"}
    )
    chain["output"] = sealed_output
    chain["output_bytes"] = canonical_resource_bytes(sealed_output)
    latest = copy.deepcopy(chain["latest_pointer"])
    latest["ledger_ref"] = _ref(sealed_ledger, chain["ledger_path"])
    latest["terminal_output_ref"] = _ref(sealed_output, chain["output_path"])
    chain["latest_pointer"] = seal_semantic(
        {key: value for key, value in latest.items() if key != "semantic_sha256"}
    )


def test_latest_pointer_binds_only_terminal_v2_ledger_and_output() -> None:
    chain = _terminal_chain()
    assert _validate_admitted_terminal_chain(chain) == chain["latest_pointer"]


def _stored_source_documents(dag: dict[str, Any]) -> dict[str, bytes]:
    return {
        **{
            path: canonical_resource_bytes(document)
            for path, document in dag["dataset_manifests"].items()
        },
        **{
            path: canonical_resource_bytes(document)
            for path, document in dag["observation_dispositions"].items()
        },
        dag["source_manifest_path"]: canonical_resource_bytes(dag["source_manifest"]),
        **{
            path: canonical_resource_bytes(document)
            for path, document in dag["generation_catalogs"].items()
        },
        **{path: canonical_resource_bytes(document) for path, document in dag["summaries"].items()},
        dag["source_binding_set_path"]: canonical_resource_bytes(dag["source_binding_set"]),
        dag["source_locator_path"]: canonical_resource_bytes(dag["source_locator"]),
    }


def test_runtime_stored_source_document_closure_is_byte_exact() -> None:
    dag = _source_dag()
    stored = _stored_source_documents(dag)
    arguments = {
        key: dag[key]
        for key in (
            "dataset_manifests",
            "observation_dispositions",
            "source_manifest",
            "source_manifest_path",
            "generation_catalogs",
            "summaries",
            "source_binding_set",
            "source_binding_set_path",
            "source_locator",
            "source_locator_path",
        )
    }
    v17_validators._validate_stored_source_document_closure(
        **arguments,
        stored_document_bytes=stored,
    )

    missing = dict(stored)
    missing.pop(dag["source_locator_path"])
    with pytest.raises(V17V2ValidationError, match="closure mismatch"):
        v17_validators._validate_stored_source_document_closure(
            **arguments,
            stored_document_bytes=missing,
        )

    noncanonical = dict(stored)
    noncanonical[dag["source_locator_path"]] = b" " + noncanonical[dag["source_locator_path"]]
    with pytest.raises(V17V2ValidationError, match="canonical compact JSON"):
        v17_validators._validate_stored_source_document_closure(
            **arguments,
            stored_document_bytes=noncanonical,
        )


def test_public_terminal_admission_fails_closed_while_registry_is_partial() -> None:
    chain = _terminal_chain()
    dag = _source_dag()
    stored_documents = _stored_source_documents(dag)
    with pytest.raises(V17V2ValidationError, match="not COMPLETE"):
        validate_shadow_terminal_chain(
            ledger_bytes=chain["ledger_bytes"],
            predecessor_ledger_bytes=chain["predecessor_ledger_bytes"],
            ledger_path=chain["ledger_path"],
            output_bytes=chain["output_bytes"],
            output_path=chain["output_path"],
            latest_pointer=chain["latest_pointer"],
            previous_pointer_bytes=chain["previous_pointer_bytes"],
            source_role_matrix=dag["source_role_matrix"],
            source_objects=dag["source_objects"],
            dataset_manifests=dag["dataset_manifests"],
            observation_dispositions=dag["observation_dispositions"],
            source_manifest=dag["source_manifest"],
            source_manifest_path=dag["source_manifest_path"],
            generation_catalogs=dag["generation_catalogs"],
            summaries=dag["summaries"],
            source_binding_set=dag["source_binding_set"],
            source_binding_set_path=dag["source_binding_set_path"],
            source_locator=dag["source_locator"],
            source_locator_path=dag["source_locator_path"],
            stored_source_document_bytes=stored_documents,
        )


@pytest.mark.parametrize("carrier", ["ledger", "output"])
def test_terminal_chain_requires_exact_canonical_stored_bytes(carrier: str) -> None:
    chain = _terminal_chain()
    chain[f"{carrier}_bytes"] = b" " + chain[f"{carrier}_bytes"]
    with pytest.raises(V17V2ValidationError, match="canonical compact JSON"):
        _validate_admitted_terminal_chain(chain)


def test_terminal_chain_rejects_fabricated_admitted_locator_hash() -> None:
    chain = _terminal_chain()
    chain["source_admission"] = replace(
        chain["source_admission"],
        locator_byte_sha256="f" * 64,
    )
    with pytest.raises(V17V2ValidationError, match="locator byte SHA-256 mismatch"):
        _validate_admitted_terminal_chain(chain)


@pytest.mark.parametrize("mutation", ["empty", "extra", "duplicate"])
def test_terminal_chain_requires_exact_nonempty_admitted_input_inventory(
    mutation: str,
) -> None:
    chain = _terminal_chain()
    inputs = chain["source_admission"].input_bindings
    if mutation == "empty":
        changed = ()
    elif mutation == "extra":
        changed = (*inputs, ("zz_extra", *inputs[0][1:]))
    else:
        changed = (*inputs, inputs[0])
    chain["source_admission"] = replace(
        chain["source_admission"],
        input_bindings=changed,
    )
    with pytest.raises(
        V17V2ValidationError,
        match="empty|do not match|not unique",
    ):
        _validate_admitted_terminal_chain(chain)


@pytest.mark.parametrize(
    ("artifact_version", "relative_path"),
    [
        (
            SHADOW_LEDGER_VERSION,
            "results/v17_shadow/protocol-v2/runs/run-1/ledger.json",
        ),
        (
            SHADOW_OUTPUT_VERSION,
            "results/v17_shadow/protocol-v2/outcomes/run-1.json",
        ),
    ],
)
def test_terminal_chain_rejects_state_carrier_artifact_cycles(
    artifact_version: str,
    relative_path: str,
) -> None:
    chain = _terminal_chain()
    ledger = copy.deepcopy(chain["ledger"])
    ledger["artifacts"] = [
        {
            "role": "state-cycle",
            "artifact_ref": {
                "artifact_id": "state-cycle",
                "artifact_version": artifact_version,
                "relative_path": relative_path,
                "byte_sha256": "a" * 64,
                "semantic_sha256": "b" * 64,
            },
            "sequence": ledger["sequence"],
            "state": ledger["state"],
        }
    ]
    ledger["history"][-1]["artifact_roles"] = ["state-cycle"]
    _install_terminal_ledger(chain, ledger)
    with pytest.raises(V17V2ValidationError, match="state-carrier artifact cycle"):
        _validate_admitted_terminal_chain(chain)


def test_shadow_ledger_timestamp_order_uses_utc_instants() -> None:
    ledger = copy.deepcopy(_terminal_chain()["ledger"])
    ledger["history"][1]["at"] = "2026-07-22T05:02:00+05:00"
    ledger = seal_semantic(
        {key: value for key, value in ledger.items() if key != "semantic_sha256"}
    )
    with pytest.raises(V17V2ValidationError, match="timestamp regressed"):
        validate_shadow_ledger(ledger)

    forward = copy.deepcopy(_terminal_chain()["ledger"])
    forward["history"][0]["at"] = "2026-07-22T05:03:00+05:00"
    forward["created_at"] = forward["history"][0]["at"]
    forward = seal_semantic(
        {key: value for key, value in forward.items() if key != "semantic_sha256"}
    )
    assert validate_shadow_ledger(forward) == forward


def test_terminal_output_and_latest_timestamps_are_causal() -> None:
    chain = _terminal_chain()
    output = copy.deepcopy(chain["output"])
    output["generated_at"] = "2026-07-22T00:06:59Z"
    chain["output_bytes"] = canonical_resource_bytes(
        seal_semantic({key: value for key, value in output.items() if key != "semantic_sha256"})
    )
    with pytest.raises(V17V2ValidationError, match="output predates"):
        _validate_admitted_terminal_chain(chain)

    chain = _terminal_chain()
    latest = copy.deepcopy(chain["latest_pointer"])
    latest["published_at"] = "2026-07-22T00:06:59Z"
    chain["latest_pointer"] = seal_semantic(
        {key: value for key, value in latest.items() if key != "semantic_sha256"}
    )
    with pytest.raises(V17V2ValidationError, match="pointer predates"):
        _validate_admitted_terminal_chain(chain)


def test_shadow_ledger_validates_independently_before_terminal_publication() -> None:
    ledger = _terminal_chain()["ledger"]
    assert validate_shadow_ledger(ledger) == ledger


def test_shadow_ledger_binds_exact_frozen_contract_and_module_inventories() -> None:
    baseline = _terminal_chain()["ledger"]
    mutations = []

    wrong_manifest = copy.deepcopy(baseline)
    wrong_manifest["contract_bindings"]["package_manifest_sha256"] = "0" * 64
    mutations.append((wrong_manifest, "package manifest binding mismatch"))

    missing_resource = copy.deepcopy(baseline)
    missing_resource["contract_bindings"]["resource_bindings"].pop()
    mutations.append((missing_resource, "resource binding inventory mismatch"))

    wrong_schema = copy.deepcopy(baseline)
    wrong_schema["contract_bindings"]["schema_bindings"][0]["byte_sha256"] = "0" * 64
    mutations.append((wrong_schema, "schema binding inventory mismatch"))

    missing_module = copy.deepcopy(baseline)
    missing_module["implementation_bindings"].pop()
    mutations.append((missing_module, "implementation binding inventory mismatch"))

    wrong_module = copy.deepcopy(baseline)
    wrong_module["implementation_bindings"][0]["byte_sha256"] = "0" * 64
    mutations.append((wrong_module, "implementation binding inventory mismatch"))

    for changed, message in mutations:
        resealed = seal_semantic(
            {key: value for key, value in changed.items() if key != "semantic_sha256"}
        )
        with pytest.raises(V17V2ValidationError, match=message):
            validate_shadow_ledger(resealed)


def test_shadow_ledger_chain_hashes_every_stored_predecessor() -> None:
    chain = _terminal_chain()
    ledger_bytes = [
        *chain["predecessor_ledger_bytes"],
        canonical_resource_bytes(chain["ledger"]),
    ]
    assert validate_shadow_ledger_chain(ledger_bytes) == chain["ledger"]

    predecessor = json.loads(chain["predecessor_ledger_bytes"][-1])
    predecessor["strategy_id"] = "tampered-shadow"
    predecessor = seal_semantic(
        {key: value for key, value in predecessor.items() if key != "semantic_sha256"}
    )
    with pytest.raises(
        V17V2ValidationError,
        match="predecessor byte SHA-256 mismatch",
    ):
        validate_shadow_ledger_successor(
            predecessor_ledger_bytes=canonical_resource_bytes(predecessor),
            successor_ledger=chain["ledger"],
        )


def test_latest_pointer_hashes_previous_pointer_bytes_for_repair() -> None:
    chain = _terminal_chain()
    previous_pointer_bytes = canonical_resource_bytes(chain["latest_pointer"])
    latest = copy.deepcopy(chain["latest_pointer"])
    latest["previous_pointer_byte_sha256"] = hashlib.sha256(previous_pointer_bytes).hexdigest()
    latest["publication_mode"] = "REPAIR"
    latest["published_at"] = "2026-07-22T00:09:00Z"
    chain["latest_pointer"] = seal_semantic(
        {key: value for key, value in latest.items() if key != "semantic_sha256"}
    )
    chain["previous_pointer_bytes"] = previous_pointer_bytes
    assert _validate_admitted_terminal_chain(chain) == chain["latest_pointer"]

    chain["previous_pointer_bytes"] = previous_pointer_bytes[:-1] + b" "
    with pytest.raises(V17V2ValidationError):
        _validate_admitted_terminal_chain(chain)


def test_shadow_ledger_rejects_n_plus_one_without_history_entry() -> None:
    ledger = copy.deepcopy(_terminal_chain()["ledger"])
    ledger["sequence"] += 1
    ledger = seal_semantic(
        {key: value for key, value in ledger.items() if key != "semantic_sha256"}
    )
    with pytest.raises(V17V2ValidationError, match="history length"):
        validate_shadow_ledger(ledger)


def test_shadow_ledger_rejects_history_input_binding_sha_tamper() -> None:
    ledger = copy.deepcopy(_terminal_chain()["ledger"])
    ledger["history"][2]["input_binding_sha256s"] = ["f" * 64]
    ledger = seal_semantic(
        {key: value for key, value in ledger.items() if key != "semantic_sha256"}
    )
    with pytest.raises(V17V2ValidationError, match="input binding SHA-256 set mismatch"):
        validate_shadow_ledger(ledger)


def test_shadow_ledger_rejects_non_total_contract_binding_order() -> None:
    ledger = copy.deepcopy(_terminal_chain()["ledger"])
    ledger["contract_bindings"]["resource_bindings"].append(
        {
            "binding_id": "alpha.v1",
            "relative_path": "resources/action_matrix.v1.json",
            "byte_sha256": "7" * 64,
        }
    )
    ledger = seal_semantic(
        {key: value for key, value in ledger.items() if key != "semantic_sha256"}
    )
    with pytest.raises(V17V2ValidationError, match="complete total order"):
        validate_shadow_ledger(ledger)


@pytest.mark.parametrize(
    "invalid_path",
    [
        "/data/private/v17_sources/protocol-v2/locators/source-locator.json",
        "data/private/v17_sources/locators/source-locator.json",
        "data/private/v17_sources/protocol-v2/locators/Source-Locator.json",
        "data/private/v17_sources/protocol-v2/locators/source:locator.json",
        "data/private/v17_sources/protocol-v2/locators/../source-locator.json",
        "data/private/v17_sources/protocol-v2/locators/./source-locator.json",
        "data/private/v17_sources/protocol-v2//locators/source-locator.json",
        "data/private/v17_sources/protocol-v2/locators/source-locator.json/",
    ],
)
def test_shadow_ledger_rejects_noncanonical_locator_paths(invalid_path: str) -> None:
    ledger = copy.deepcopy(_terminal_chain()["ledger"])
    ledger["locator_binding"]["locator_ref"]["relative_path"] = invalid_path
    ledger = seal_semantic(
        {key: value for key, value in ledger.items() if key != "semantic_sha256"}
    )
    with pytest.raises(V17V2ValidationError, match="does not match pattern"):
        validate_shadow_ledger(ledger)


def _failure_receipt() -> dict[str, Any]:
    return _sealed(
        version=ACTION_FAILURE_RECEIPT_VERSION,
        receipt_id="failure-1",
        receipt_path=("results/v17_shadow/protocol-v2/runs/run-1/receipts/failure-1.json"),
        run_id="run-1",
        action="SHADOW_RECEIVE",
        acceptance_checkpoint="INITIALIZED",
        status="UNPUBLISHED_NOT_COMMITTED",
        reason_code="validation-failed",
        detail="post-initialized validation failed",
        expected_ledger_sha256="a" * 64,
        observed_ledger_sha256=None,
        durably_committed=False,
        write_effect="RECEIPT_ONLY",
        created_at="2026-07-22T00:06:00Z",
    )


def test_action_failure_receipt_is_post_initialized_only() -> None:
    receipt = _failure_receipt()
    assert validate_action_failure_receipt(receipt) == receipt
    changed = copy.deepcopy(receipt)
    changed["acceptance_checkpoint"] = "PRE_IMPORT"
    changed = seal_semantic(
        {key: value for key, value in changed.items() if key != "semantic_sha256"}
    )
    with pytest.raises(V17V2ValidationError, match="does not match const"):
        validate_action_failure_receipt(changed)


def test_action_failure_receipt_rejects_zero_write_rejection_receipt() -> None:
    receipt = copy.deepcopy(_failure_receipt())
    receipt["status"] = "REJECTED_ZERO_WRITE"
    receipt = seal_semantic(
        {key: value for key, value in receipt.items() if key != "semantic_sha256"}
    )
    with pytest.raises(V17V2ValidationError, match="outside enum"):
        validate_action_failure_receipt(receipt)


def test_latest_pointer_rejects_nonterminal_ledger() -> None:
    chain = _terminal_chain()
    ledger = copy.deepcopy(chain["ledger"])
    ledger["state"] = "DEEP_RESPONSE_RECEIVED"
    ledger["sequence"] = 3
    ledger["action"] = "SHADOW_RECEIVE"
    ledger["updated_at"] = ledger["history"][3]["at"]
    ledger["previous_ledger_sha256"] = ledger["history"][3]["expected_ledger_sha256"]
    ledger["history"] = ledger["history"][:4]
    ledger = seal_semantic(
        {key: value for key, value in ledger.items() if key != "semantic_sha256"}
    )
    chain["ledger"] = ledger
    chain["ledger_bytes"] = canonical_resource_bytes(ledger)
    chain["predecessor_ledger_bytes"] = chain["predecessor_ledger_bytes"][:3]
    with pytest.raises(V17V2ValidationError, match="not terminal"):
        _validate_admitted_terminal_chain(chain)
