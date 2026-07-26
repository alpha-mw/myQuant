from __future__ import annotations

from dataclasses import replace
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

import pytest

from quant_investor.v17_v2_contract.canonical import (
    canonical_resource_bytes,
    load_canonical_resource,
)
from quant_investor.v17_v2_contract.resources import load_packaged_json
from quant_investor.v17_v2_runtime import sources as subject
from quant_investor.v17_v2_runtime.sources import (
    DatasetRecordSpec,
    DatasetShardFacts,
    SourceFile,
    SourcePlanningError,
    plan_source_dag,
)

CUTOFF = "2026-07-26T07:00:00Z"
CREATED_AT = "2026-07-26T07:01:00Z"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _complete_matrix() -> dict[str, Any]:
    return {
        "protocol_version": "myquant.v17.v2",
        "version": "myquant.v17.v2.source-role-matrix.v1",
        "authority": False,
        "completeness": "COMPLETE",
        "runtime_usable": True,
        "forbidden_role_suffixes": ["_verification_receipt"],
        "ordering": {
            "pending_registry": "ascii_casefold_ascending",
            "roles": "role_ascii_casefold_ascending",
        },
        "pending_registry": [],
        "conditional_semantics": [],
        "roles": [
            {
                "role": "market_bars_dataset",
                "phase": "RANK",
                "kind": "DATASET",
                "required": True,
                "availability_disposition": (
                    "REJECT_BEFORE_INITIALIZED_ZERO_WRITE"
                ),
                "schema_status": "FROZEN",
                "schema_version": "myquant.v17.v2.dataset-manifest.schema.v1",
            },
            {
                "role": "pit_generation_catalog",
                "phase": "RANK",
                "kind": "OBJECT",
                "required": True,
                "availability_disposition": (
                    "REJECT_BEFORE_INITIALIZED_ZERO_WRITE"
                ),
                "schema_status": "FROZEN",
                "schema_version": (
                    "myquant.v17.v2.generation-catalog.schema.v1"
                ),
            },
        ],
    }


def _record_registry() -> dict[str, Any]:
    return {
        "version": "fixture.dataset-record-registry.v1",
        "records": [{"role": "market_bars_dataset"}],
    }


def _record_spec(
    role: str,
    registry: Mapping[str, Any],
) -> DatasetRecordSpec:
    assert role == "market_bars_dataset"
    assert registry == _record_registry()
    return DatasetRecordSpec(
        catalog_role="pit_generation_catalog",
        record_schema_id="market-bars-dataset.v1",
        schema=(
            {
                "name": "trade_date",
                "logical_type": "date",
                "nullable": False,
            },
        ),
        primary_key=("trade_date",),
        partition_keys=(),
        sort_keys=("trade_date",),
        valid_time_field="trade_date",
        available_time_field="trade_date",
    )


def _inspect(
    source: SourceFile,
    raw: bytes,
    spec: DatasetRecordSpec,
) -> DatasetShardFacts:
    assert source.role == "market_bars_dataset"
    assert raw.startswith(b"PAR1")
    assert spec.record_schema_id == "market-bars-dataset.v1"
    return DatasetShardFacts(
        logical_name="market-bars.parquet",
        partition_values={},
        row_count=1,
        min_key=("2026-07-25",),
        max_key=("2026-07-25",),
        observation_key_sha256s=(_sha(b"2026-07-25"),),
    )


def _fixture_sources(root: Path) -> tuple[SourceFile, SourceFile]:
    dataset_raw = b"PAR1exact-dataset-bytes"
    catalog_raw = b'{"generation":"fixture"}\n'
    dataset_path = root / "market.parquet"
    catalog_path = root / "pit.json"
    dataset_path.write_bytes(dataset_raw)
    catalog_path.write_bytes(catalog_raw)
    return (
        SourceFile(
            path=dataset_path,
            expected_sha256=_sha(dataset_raw),
            role="market_bars_dataset",
        ),
        SourceFile(
            path=catalog_path,
            expected_sha256=_sha(catalog_raw),
            role="pit_generation_catalog",
        ),
    )


def _fake_dag_validator(**dag: Any) -> Mapping[str, Any]:
    assert dag["dataset_record_schema_registry"] == _record_registry()
    assert set(dag["source_objects"]) == {
        row["source_ref"]["relative_path"]
        for row in dag["source_manifest"]["sources"]
    }
    assert len(dag["dataset_manifests"]) == 1
    assert len(dag["observation_dispositions"]) == 1
    assert len(dag["generation_catalogs"]) == 1
    assert len(dag["summaries"]) == 1
    assert len(dag["source_binding_set"]["bindings"]) == 1
    return dag["source_locator"]


def _plan(root: Path, source_values: tuple[SourceFile, ...] | None = None):
    sources = source_values or _fixture_sources(root)
    return plan_source_dag(
        source_root=root,
        sources=sources,
        cutoff=CUTOFF,
        created_at=CREATED_AT,
        source_role_matrix=_complete_matrix(),
        dataset_record_schema_registry=_record_registry(),
        dataset_inspector=_inspect,
        role_matrix_validator=lambda value: value,
        record_registry_validator=lambda value: value,
        record_spec_resolver=_record_spec,
        dag_validator=_fake_dag_validator,
    )


def test_plans_exact_canonical_bytes_without_writing_destinations(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "inputs"
    source_root.mkdir()
    plan = _plan(source_root)

    assert plan.write_intents[-1].kind == "SOURCE_LOCATOR"
    assert [intent.sequence for intent in plan.write_intents] == sorted(
        intent.sequence for intent in plan.write_intents
    )
    assert len({intent.relative_path.casefold() for intent in plan.write_intents}) == len(
        plan.write_intents
    )
    for intent in plan.write_intents:
        assert _sha(intent.payload) == intent.byte_sha256
        assert not (tmp_path / intent.relative_path).exists()
        if intent.kind != "SOURCE_OBJECT":
            assert canonical_resource_bytes(load_canonical_resource(intent.payload)) == (
                intent.payload
            )
    object_intents = {
        intent.byte_sha256: intent.payload
        for intent in plan.write_intents
        if intent.kind == "SOURCE_OBJECT"
    }
    assert object_intents == {
        source.expected_sha256: source.path.read_bytes()
        for source in _fixture_sources(source_root)
    }


def test_plan_passes_packaged_registry_and_real_dag_cross_validator(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "inputs"
    source_root.mkdir()
    sources = _fixture_sources(source_root)
    registry = load_packaged_json(
        "resources/dataset_record_schema_registry.v1.json"
    )

    def inspect_packaged_record(
        source: SourceFile,
        raw: bytes,
        spec: DatasetRecordSpec,
    ) -> DatasetShardFacts:
        assert source.role == "market_bars_dataset"
        assert raw.startswith(b"PAR1")
        assert spec.record_schema_id == "market-bars-dataset.v1"
        return DatasetShardFacts(
            logical_name="trade_date=2026-07-25/market-bars.parquet",
            partition_values={"trade_date": "2026-07-25"},
            row_count=1,
            min_key=("000001.SZ", "2026-07-25"),
            max_key=("000001.SZ", "2026-07-25"),
            observation_key_sha256s=(_sha(b"000001.SZ|2026-07-25"),),
        )

    plan = plan_source_dag(
        source_root=source_root,
        sources=sources,
        cutoff=CUTOFF,
        created_at=CREATED_AT,
        source_role_matrix=_complete_matrix(),
        dataset_record_schema_registry=registry,
        dataset_inspector=inspect_packaged_record,
        role_matrix_validator=lambda value: value,
    )
    assert plan.source_locator_path.endswith(f"/{plan.source_locator['locator_id']}.json")


def test_partial_matrix_rejects_before_any_source_read(tmp_path: Path) -> None:
    missing = SourceFile(
        path=tmp_path / "missing.parquet",
        expected_sha256="0" * 64,
        role="market_bars_dataset",
    )
    partial = {**_complete_matrix(), "completeness": "PARTIAL", "runtime_usable": False}
    with pytest.raises(SourcePlanningError, match="must be COMPLETE"):
        plan_source_dag(
            source_root=tmp_path,
            sources=(missing,),
            cutoff=CUTOFF,
            created_at=CREATED_AT,
            source_role_matrix=partial,
            dataset_record_schema_registry=_record_registry(),
            dataset_inspector=_inspect,
            role_matrix_validator=lambda value: value,
            record_registry_validator=lambda value: value,
            record_spec_resolver=_record_spec,
            dag_validator=_fake_dag_validator,
        )


def test_roles_and_content_objects_are_exactly_one_to_one(tmp_path: Path) -> None:
    source_root = tmp_path / "inputs"
    source_root.mkdir()
    dataset, catalog = _fixture_sources(source_root)
    duplicate_role = replace(catalog, role=dataset.role)
    with pytest.raises(SourcePlanningError, match="casefold collision"):
        _plan(source_root, (dataset, duplicate_role))

    shared_path = replace(
        catalog,
        path=dataset.path,
        expected_sha256=dataset.expected_sha256,
    )
    with pytest.raises(SourcePlanningError, match="paths have an exact"):
        _plan(source_root, (dataset, shared_path))


@pytest.mark.parametrize("attack", ["symlink", "hardlink"])
def test_secure_source_reader_rejects_symlink_and_hardlink(
    tmp_path: Path,
    attack: str,
) -> None:
    source_root = tmp_path / "inputs"
    source_root.mkdir()
    dataset, catalog = _fixture_sources(source_root)
    attacked = source_root / "attacked.parquet"
    if attack == "symlink":
        attacked.symlink_to(dataset.path)
    else:
        os.link(dataset.path, attacked)
    attacked_source = replace(dataset, path=attacked)
    with pytest.raises(SourcePlanningError, match="symlink|hardlinked"):
        _plan(source_root, (attacked_source, catalog))


def test_secure_source_reader_rejects_path_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "inputs"
    source_root.mkdir()
    dataset, catalog = _fixture_sources(source_root)
    replacement = source_root / "replacement.parquet"
    replacement.write_bytes(dataset.path.read_bytes())
    original_chain = subject._secure_chain
    calls = 0

    def swapping_chain(path: str):
        nonlocal calls
        if path == str(dataset.path):
            calls += 1
            if calls == 2:
                displaced = source_root / "displaced.parquet"
                os.replace(dataset.path, displaced)
                os.replace(replacement, dataset.path)
        return original_chain(path)

    monkeypatch.setattr(subject, "_secure_chain", swapping_chain)
    with pytest.raises(SourcePlanningError, match="path changed"):
        _plan(source_root, (dataset, catalog))


def test_expected_sha_and_csv_fallback_fail_closed(tmp_path: Path) -> None:
    source_root = tmp_path / "inputs"
    source_root.mkdir()
    dataset, catalog = _fixture_sources(source_root)
    wrong_sha = replace(dataset, expected_sha256="f" * 64)
    with pytest.raises(SourcePlanningError, match="SHA-256 mismatch"):
        _plan(source_root, (wrong_sha, catalog))

    csv_path = source_root / "market.csv"
    csv_path.write_bytes(dataset.path.read_bytes())
    csv_source = replace(dataset, path=csv_path)
    with pytest.raises(SourcePlanningError, match="CSV fallback is forbidden"):
        _plan(source_root, (csv_source, catalog))
