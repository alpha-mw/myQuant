from __future__ import annotations

import hashlib
from pathlib import Path
import re
import stat

import pytest

from quant_investor.v17.contracts import V17ContractError
from quant_investor.v17.resources import (
    FROZEN_OPERATIONAL_RESOURCE_SHA256S,
    FROZEN_POLICY_RESOURCE_SHA256S,
    FROZEN_SCHEMA_SHA256S,
    RESOURCE_DIR,
    SCHEMA_DIR,
    assert_supported_json_schema,
    assert_frozen_package_contracts,
    load_json_schema,
    schema_byte_sha256,
)
from quant_investor.v17.semantic import (
    canonical_json_bytes,
    seal_semantic,
    semantic_sha256,
    validate_semantic_seal,
)
from quant_investor.v17.storage import (
    atomic_write_json,
    ensure_v17_shadow_layout,
    read_json,
)


def test_semantic_sha_is_compact_sorted_utf8_without_self_field() -> None:
    left = {"中文": "值", "a": [1, 2], "semantic_sha256": "0" * 64}
    right = {"a": [1, 2], "中文": "值"}
    expected = hashlib.sha256(b'{"a":[1,2],"\xe4\xb8\xad\xe6\x96\x87":"\xe5\x80\xbc"}').hexdigest()
    assert canonical_json_bytes(right) == ('{"a":[1,2],"中文":"值"}'.encode("utf-8"))
    assert semantic_sha256(left) == semantic_sha256(right) == expected
    assert validate_semantic_seal(seal_semantic(right))["semantic_sha256"] == expected


def test_atomic_write_is_0600_and_parent_is_0700(tmp_path: Path) -> None:
    root = tmp_path / "private"
    target = root / "nested" / "artifact.json"
    digest = atomic_write_json(target, {"safe": True}, root=root)
    assert digest == hashlib.sha256(b'{"safe":true}\n').hexdigest()
    assert read_json(target) == {"safe": True}
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(root.stat().st_mode) == 0o700


def test_v17_layout_never_chmods_repository_or_shared_parents(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    results = repo / "results"
    data = repo / "data"
    results.mkdir(parents=True, mode=0o755)
    data.mkdir(mode=0o755)
    repo.chmod(0o755)
    results.chmod(0o755)
    data.chmod(0o755)

    layout = ensure_v17_shadow_layout(repo)

    assert stat.S_IMODE(repo.stat().st_mode) == 0o755
    assert stat.S_IMODE(results.stat().st_mode) == 0o755
    assert stat.S_IMODE(data.stat().st_mode) == 0o755
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o700 for path in layout.values())


def test_storage_rejects_symlink_and_duplicate_json(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    outside = tmp_path / "outside.json"
    outside.write_text('{"safe":true}', encoding="utf-8")
    link = root / "link.json"
    link.symlink_to(outside)
    with pytest.raises(V17ContractError, match="symlink"):
        read_json(link)

    duplicate = root / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}', encoding="utf-8")
    with pytest.raises(V17ContractError, match="duplicate"):
        read_json(duplicate)


@pytest.mark.parametrize(
    "schema_name",
    sorted(path.name for path in SCHEMA_DIR.glob("*.json")),
)
def test_packaged_schemas_are_hash_bound_and_valid(schema_name: str) -> None:
    digest = schema_byte_sha256(schema_name)
    schema = load_json_schema(schema_name, expected_sha256=digest)
    assert schema["$schema"].endswith("2020-12/schema")
    with pytest.raises(V17ContractError, match="mismatch"):
        load_json_schema(schema_name, expected_sha256="0" * 64)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"unsupportedKeyword": True}, "unsupported JSON Schema keywords"),
        ({"$ref": "#/$defs/missing"}, "unresolved JSON Schema reference"),
        ({"pattern": "["}, "invalid JSON Schema pattern"),
    ],
)
def test_supported_schema_gate_rejects_malformed_constructs(
    mutation: dict[str, object],
    message: str,
) -> None:
    schema: dict[str, object] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://myquant.local/test.schema.json",
        "type": "object",
        "$defs": {},
    }
    schema.update(mutation)
    with pytest.raises(V17ContractError, match=message):
        assert_supported_json_schema(schema)


def test_corrective_v2_schema_package_is_frozen_by_exact_byte_sha() -> None:
    expected = {
        "dataset_manifest.v1.schema.json": (
            "46372f960ea9baf424f55d3abc603c8c171e95933f72e64c06fb52ada315061e"
        ),
        "deep_research_request.v2.schema.json": (
            "e690e320aa1c9afaec5407a6beb94037609604339133a336721ad1e83de6a7ab"
        ),
        "deep_research_response.v2.schema.json": (
            "a33ce2171f0a2624d6fa861b26d7ed12282582606a97550b953d92de0c9d65cc"
        ),
        "generation_catalog.v1.schema.json": (
            "85eaf777c72bc5ef867275e5a8f494a81ff84fc632cc1777516c1d9f0a9be4c8"
        ),
        "observation_disposition.v1.schema.json": (
            "49f8d80550772f376e5beaa63e814657c5f4340e5418c81a8268e27370f73cbe"
        ),
        "source_manifest.v2.schema.json": (
            "e11bad313af50e7453265912fadad472c3c092c03e4d1d02b69db7b9ac773b92"
        ),
    }
    assert {name: FROZEN_SCHEMA_SHA256S[name] for name in expected} == expected
    assert {name: schema_byte_sha256(name) for name in expected} == expected
    observed = assert_frozen_package_contracts()
    assert observed["schemas"] == FROZEN_SCHEMA_SHA256S


def test_frozen_resource_inventory_exactly_covers_packaged_json() -> None:
    observed = assert_frozen_package_contracts()
    assert observed["resources"] == FROZEN_POLICY_RESOURCE_SHA256S
    assert observed["operational_resources"] == FROZEN_OPERATIONAL_RESOURCE_SHA256S
    assert {path.name for path in RESOURCE_DIR.glob("*.json")} == set(
        FROZEN_POLICY_RESOURCE_SHA256S
    ) | set(FROZEN_OPERATIONAL_RESOURCE_SHA256S)


def test_source_manifest_v2_freezes_strict_object_dataset_tagged_union() -> None:
    schema = load_json_schema(
        "source_manifest.v2.schema.json",
        expected_sha256=FROZEN_SCHEMA_SHA256S["source_manifest.v2.schema.json"],
    )
    assert schema["additionalProperties"] is False
    assert schema["properties"]["authority"] == {"const": False}
    branches = schema["properties"]["sources"]["items"]["oneOf"]
    assert branches == [
        {"$ref": "#/$defs/availableJsonObject"},
        {"$ref": "#/$defs/availableDataset"},
        {"$ref": "#/$defs/unavailableSource"},
    ]
    assert schema["$defs"]["availableJsonObject"]["properties"]["kind"] == {"const": "OBJECT"}
    dataset = schema["$defs"]["availableDataset"]
    assert dataset["additionalProperties"] is False
    assert dataset["properties"]["kind"] == {"const": "DATASET"}
    assert "dataset_manifest_ref" in dataset["required"]
    assert "object_ref" not in dataset["properties"]


def test_dataset_catalog_disposition_and_deep_v2_shapes_are_strict() -> None:
    dataset = load_json_schema(
        "dataset_manifest.v1.schema.json",
        expected_sha256=FROZEN_SCHEMA_SHA256S["dataset_manifest.v1.schema.json"],
    )
    assert dataset["additionalProperties"] is False
    assert dataset["properties"]["format"] == {"enum": ["PARQUET", "BLOB"]}
    assert dataset["$defs"]["schemaEntry"]["additionalProperties"] is False
    assert dataset["$defs"]["shard"]["additionalProperties"] is False
    logical_name_pattern = dataset["$defs"]["shard"]["properties"]["logical_name"]["pattern"]
    assert re.fullmatch(logical_name_pattern, "partition/date=2026-07-22/part-000.parquet")
    for invalid in ("partition//part", "partition/./part", "partition/../part"):
        assert re.fullmatch(logical_name_pattern, invalid) is None
    assert dataset["$defs"]["shard"]["properties"]["size_bytes"] == {
        "type": "integer",
        "minimum": 1,
        "maximum": 8589934592,
    }
    assert dataset["$defs"]["finiteScalar"]["type"] == [
        "string",
        "integer",
        "number",
        "boolean",
    ]
    parquet = dataset["allOf"][0]["then"]["properties"]
    parquet_shard = parquet["shards"]["items"]["properties"]
    assert parquet_shard["row_count"] == {"type": "integer", "minimum": 1}
    assert parquet_shard["min_key"]["type"] == "array"
    assert parquet_shard["max_key"]["type"] == "array"
    blob = dataset["allOf"][0]["else"]["properties"]
    assert blob["total_row_count"] == {"const": 0}
    blob_shard = blob["shards"]["items"]["properties"]
    assert blob_shard["partition_values"] == {"maxProperties": 0}
    assert blob_shard["row_count"] == {"const": 0}
    assert blob_shard["min_key"] == {"const": None}
    assert blob_shard["max_key"] == {"const": None}

    catalog = load_json_schema(
        "generation_catalog.v1.schema.json",
        expected_sha256=FROZEN_SCHEMA_SHA256S["generation_catalog.v1.schema.json"],
    )
    assert catalog["additionalProperties"] is False
    assert catalog["properties"]["storage_scope"] == {"const": "V17_PRIVATE"}
    assert catalog["$defs"]["table"]["additionalProperties"] is False

    dispositions = load_json_schema(
        "observation_disposition.v1.schema.json",
        expected_sha256=FROZEN_SCHEMA_SHA256S["observation_disposition.v1.schema.json"],
    )
    item = dispositions["$defs"]["disposition"]
    assert item["additionalProperties"] is False
    assert item["properties"]["status"] == {"enum": ["INVALID", "UNAVAILABLE", "UNREADY"]}
    assert len(item["allOf"]) == 3
    assert "dispositions" not in dispositions["properties"]
    assert dispositions["properties"]["status_counts"]["additionalProperties"] is False
    assert dispositions["properties"]["status_counts"]["required"] == [
        "INVALID",
        "UNAVAILABLE",
        "UNREADY",
    ]
    assert dispositions["properties"]["dataset_manifest_ref"] == {
        "$ref": "#/$defs/datasetManifestRef"
    }
    disposition_ref = dispositions["$defs"]["datasetManifestRef"]
    assert disposition_ref["additionalProperties"] is False
    assert disposition_ref["properties"]["schema_version"] == {
        "const": "myquant.v17.dataset-manifest.v1"
    }
    assert "record_schema_sha256" in dispositions["required"]
    assert "content_set_sha256" in dispositions["required"]

    for name, version in (
        ("deep_research_request.v2.schema.json", "myquant.v17.deep-research-request.v2"),
        ("deep_research_response.v2.schema.json", "myquant.v17.deep-research-response.v2"),
    ):
        deep = load_json_schema(name, expected_sha256=FROZEN_SCHEMA_SHA256S[name])
        assert deep["additionalProperties"] is False
        assert deep["properties"]["version"] == {"const": version}
        assert deep["properties"]["authority"] == {"const": False}
