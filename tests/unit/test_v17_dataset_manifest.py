from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
from typing import Any

import pytest

from quant_investor.v17.dataset_manifest import (
    V17DatasetManifestError,
    dataset_content_set_sha256,
    dataset_schema_sha256,
    derive_dataset_object_path,
    validate_dataset_manifest,
    validate_dataset_objects,
)
from quant_investor.v17.semantic import seal_semantic

PARQUET_SCHEMA = [
    {"name": "trade_date", "logical_type": "string", "nullable": False},
    {"name": "ts_code", "logical_type": "string", "nullable": False},
    {"name": "close", "logical_type": "double", "nullable": True},
]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_object(repo: Path, object_path: str, payload: bytes) -> Path:
    path = repo.joinpath(*Path(object_path).parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(0o600)
    return path


def _require_pyarrow() -> tuple[Any, Any]:
    pa = pytest.importorskip("pyarrow", reason="Parquet manifest readback requires pyarrow")
    pq = pytest.importorskip(
        "pyarrow.parquet",
        reason="Parquet manifest readback requires pyarrow.parquet",
    )
    return pa, pq


def _blob_manifest(shards: list[dict[str, Any]]) -> dict[str, Any]:
    normalized = copy.deepcopy(shards)
    payload = {
        "version": "myquant.v17.dataset-manifest.v1",
        "dataset_id": "blob_dataset",
        "role": "deep_evidence",
        "format": "BLOB",
        "media_type": "application/octet-stream",
        "schema": [],
        "primary_key": [],
        "partition_keys": [],
        "sort_keys": [],
        "shards": normalized,
        "total_row_count": sum(int(item["row_count"]) for item in normalized),
        "total_size_bytes": sum(int(item["size_bytes"]) for item in normalized),
        "content_set_sha256": dataset_content_set_sha256(normalized),
        "authority": False,
    }
    return seal_semantic(payload)


def _blob_shard(payload: bytes, *, logical_name: str = "raw/evidence.blob") -> dict[str, Any]:
    digest = _sha256(payload)
    return {
        "logical_name": logical_name,
        "partition_values": {},
        "object_path": derive_dataset_object_path(digest, "BLOB"),
        "byte_sha256": digest,
        "size_bytes": len(payload),
        "row_count": 0,
        "min_key": None,
        "max_key": None,
        "schema_sha256": dataset_schema_sha256([]),
    }


def _parquet_shard(
    repo: Path,
    rows: list[dict[str, Any]],
    *,
    logical_name: str,
    min_key: list[Any],
    max_key: list[Any],
) -> dict[str, Any]:
    pa, pq = _require_pyarrow()
    table = pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                pa.field("trade_date", pa.string(), nullable=False),
                pa.field("ts_code", pa.string(), nullable=False),
                pa.field("close", pa.float64(), nullable=True),
            ]
        ),
    )
    temporary = repo / f"{logical_name.replace('/', '_')}.parquet"
    pq.write_table(table, temporary)
    raw = temporary.read_bytes()
    digest = _sha256(raw)
    object_path = derive_dataset_object_path(digest, "PARQUET")
    _write_object(repo, object_path, raw)
    temporary.unlink()
    return {
        "logical_name": logical_name,
        "partition_values": {"trade_date": rows[0]["trade_date"]},
        "object_path": object_path,
        "byte_sha256": digest,
        "size_bytes": len(raw),
        "row_count": len(rows),
        "min_key": min_key,
        "max_key": max_key,
        "schema_sha256": dataset_schema_sha256(PARQUET_SCHEMA),
    }


def _parquet_manifest(shards: list[dict[str, Any]]) -> dict[str, Any]:
    normalized = copy.deepcopy(shards)
    payload = {
        "version": "myquant.v17.dataset-manifest.v1",
        "dataset_id": "market_snapshot",
        "role": "fundamental_history",
        "format": "PARQUET",
        "media_type": "application/vnd.apache.parquet",
        "schema": PARQUET_SCHEMA,
        "primary_key": ["ts_code"],
        "partition_keys": ["trade_date"],
        "sort_keys": ["ts_code"],
        "shards": normalized,
        "total_row_count": sum(int(item["row_count"]) for item in normalized),
        "total_size_bytes": sum(int(item["size_bytes"]) for item in normalized),
        "content_set_sha256": dataset_content_set_sha256(normalized),
        "authority": False,
    }
    return seal_semantic(payload)


def test_valid_blob_manifest_validates_and_derives_object_path(tmp_path: Path) -> None:
    payload = b"raw blob evidence"
    shard = _blob_shard(payload)
    _write_object(tmp_path, shard["object_path"], payload)

    manifest = validate_dataset_objects(_blob_manifest([shard]), repo_root=tmp_path)

    assert manifest["format"] == "BLOB"
    assert manifest["shards"][0]["object_path"] == derive_dataset_object_path(
        shard["byte_sha256"], "BLOB"
    )


def test_valid_parquet_manifest_reads_actual_schema_and_row_count(tmp_path: Path) -> None:
    shard = _parquet_shard(
        tmp_path,
        [
            {"trade_date": "2026-07-21", "ts_code": "000001.SZ", "close": 10.0},
            {"trade_date": "2026-07-21", "ts_code": "000002.SZ", "close": 11.0},
        ],
        logical_name="2026-07-21/part-000.parquet",
        min_key=["000001.SZ"],
        max_key=["000002.SZ"],
    )

    manifest = validate_dataset_objects(_parquet_manifest([shard]), repo_root=tmp_path)

    assert manifest["total_row_count"] == 2
    assert manifest["schema"] == PARQUET_SCHEMA


def test_parquet_declared_key_bounds_must_match_actual_rows(tmp_path: Path) -> None:
    shard = _parquet_shard(
        tmp_path,
        [{"trade_date": "2026-07-21", "ts_code": "999999.SZ", "close": 10.0}],
        logical_name="2026-07-21/part-000.parquet",
        min_key=["000001.SZ"],
        max_key=["000001.SZ"],
    )

    with pytest.raises(V17DatasetManifestError, match="actual min_key mismatch"):
        validate_dataset_objects(_parquet_manifest([shard]), repo_root=tmp_path)


def test_parquet_rows_must_match_declared_partition_values(tmp_path: Path) -> None:
    shard = _parquet_shard(
        tmp_path,
        [{"trade_date": "2026-07-21", "ts_code": "000001.SZ", "close": 10.0}],
        logical_name="2026-07-21/part-000.parquet",
        min_key=["000001.SZ"],
        max_key=["000001.SZ"],
    )
    shard["partition_values"] = {"trade_date": "2026-07-22"}

    with pytest.raises(V17DatasetManifestError, match="partition_values"):
        validate_dataset_objects(_parquet_manifest([shard]), repo_root=tmp_path)


def test_parquet_rows_must_follow_declared_sort_keys(tmp_path: Path) -> None:
    shard = _parquet_shard(
        tmp_path,
        [
            {"trade_date": "2026-07-21", "ts_code": "000002.SZ", "close": 11.0},
            {"trade_date": "2026-07-21", "ts_code": "000001.SZ", "close": 10.0},
        ],
        logical_name="2026-07-21/part-000.parquet",
        min_key=["000001.SZ"],
        max_key=["000002.SZ"],
    )

    with pytest.raises(V17DatasetManifestError, match="sort_keys"):
        validate_dataset_objects(_parquet_manifest([shard]), repo_root=tmp_path)


def test_schema_content_and_semantic_digests_are_domain_bound(tmp_path: Path) -> None:
    shard = _blob_shard(b"digest payload")
    manifest = _blob_manifest([shard])

    assert dataset_schema_sha256([]) == shard["schema_sha256"]
    assert dataset_content_set_sha256(manifest["shards"]) == manifest["content_set_sha256"]
    assert manifest["semantic_sha256"] != manifest["content_set_sha256"]


def test_reordered_shards_are_rejected(tmp_path: Path) -> None:
    early = _parquet_shard(
        tmp_path,
        [{"trade_date": "2026-07-21", "ts_code": "000001.SZ", "close": 10.0}],
        logical_name="2026-07-21/a.parquet",
        min_key=["000001.SZ"],
        max_key=["000001.SZ"],
    )
    late = _parquet_shard(
        tmp_path,
        [{"trade_date": "2026-07-22", "ts_code": "000002.SZ", "close": 11.0}],
        logical_name="2026-07-22/b.parquet",
        min_key=["000002.SZ"],
        max_key=["000002.SZ"],
    )

    with pytest.raises(V17DatasetManifestError, match="canonical deterministic order"):
        validate_dataset_manifest(_parquet_manifest([late, early]))


def test_same_partition_overlapping_key_ranges_are_rejected(tmp_path: Path) -> None:
    left = _parquet_shard(
        tmp_path,
        [{"trade_date": "2026-07-21", "ts_code": "000001.SZ", "close": 10.0}],
        logical_name="2026-07-21/a.parquet",
        min_key=["000001.SZ"],
        max_key=["000003.SZ"],
    )
    right = _parquet_shard(
        tmp_path,
        [{"trade_date": "2026-07-21", "ts_code": "000003.SZ", "close": 11.0}],
        logical_name="2026-07-21/b.parquet",
        min_key=["000003.SZ"],
        max_key=["000004.SZ"],
    )

    with pytest.raises(V17DatasetManifestError, match="overlap"):
        validate_dataset_manifest(_parquet_manifest([left, right]))


def test_wrong_total_row_count_is_rejected(tmp_path: Path) -> None:
    manifest = _blob_manifest([_blob_shard(b"payload")])
    manifest["total_row_count"] = 1
    manifest = seal_semantic({k: v for k, v in manifest.items() if k != "semantic_sha256"})

    with pytest.raises(V17DatasetManifestError, match="total_row_count"):
        validate_dataset_manifest(manifest)


def test_wrong_content_set_digest_is_rejected(tmp_path: Path) -> None:
    manifest = _blob_manifest([_blob_shard(b"payload")])
    manifest["content_set_sha256"] = "0" * 64
    manifest = seal_semantic({k: v for k, v in manifest.items() if k != "semantic_sha256"})

    with pytest.raises(V17DatasetManifestError, match="content_set_sha256 mismatch"):
        validate_dataset_manifest(manifest)


def test_shard_logical_name_path_traversal_is_rejected(tmp_path: Path) -> None:
    shard = _blob_shard(b"payload", logical_name="../escape.blob")

    with pytest.raises(V17DatasetManifestError, match="canonical relative logical name"):
        validate_dataset_manifest(_blob_manifest([shard]))


def test_manifest_object_path_must_be_derived_from_byte_digest(tmp_path: Path) -> None:
    shard = _blob_shard(b"payload")
    shard["object_path"] = "data/private/v17_sources/objects/../escape.blob"

    with pytest.raises(V17DatasetManifestError, match="object_path is not derived"):
        validate_dataset_manifest(_blob_manifest([shard]))


def test_duplicate_physical_object_cannot_be_counted_as_two_logical_shards(
    tmp_path: Path,
) -> None:
    first = _blob_shard(b"payload", logical_name="raw/a.blob")
    second = copy.deepcopy(first)
    second["logical_name"] = "raw/b.blob"

    with pytest.raises(V17DatasetManifestError, match="duplicate shard object_path"):
        validate_dataset_manifest(_blob_manifest([first, second]))


def test_shard_size_above_streaming_limit_is_rejected(tmp_path: Path) -> None:
    shard = _blob_shard(b"payload")
    shard["size_bytes"] = 8 * 1024 * 1024 * 1024 + 1

    with pytest.raises(V17DatasetManifestError, match="size_bytes exceeds fixed limit"):
        validate_dataset_manifest(_blob_manifest([shard]))


def test_missing_dataset_object_is_rejected(tmp_path: Path) -> None:
    shard = _blob_shard(b"payload")

    with pytest.raises(V17DatasetManifestError, match="dataset object unavailable"):
        validate_dataset_objects(_blob_manifest([shard]), repo_root=tmp_path)


def test_tampered_dataset_object_is_rejected(tmp_path: Path) -> None:
    shard = _blob_shard(b"payload")
    _write_object(tmp_path, shard["object_path"], b"tampered")

    with pytest.raises(V17DatasetManifestError, match="size mismatch|byte SHA mismatch"):
        validate_dataset_objects(_blob_manifest([shard]), repo_root=tmp_path)


def test_dataset_object_symlink_is_rejected(tmp_path: Path) -> None:
    shard = _blob_shard(b"payload")
    path = tmp_path.joinpath(*Path(shard["object_path"]).parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.blob"
    outside.write_bytes(b"payload")
    path.symlink_to(outside)

    with pytest.raises(V17DatasetManifestError, match="identity invalid"):
        validate_dataset_objects(_blob_manifest([shard]), repo_root=tmp_path)


def test_dataset_object_hardlink_is_rejected(tmp_path: Path) -> None:
    shard = _blob_shard(b"payload")
    path = _write_object(tmp_path, shard["object_path"], b"payload")
    os.link(path, tmp_path / "object-hardlink.blob")

    with pytest.raises(V17DatasetManifestError, match="identity invalid"):
        validate_dataset_objects(_blob_manifest([shard]), repo_root=tmp_path)


def test_parquet_row_count_mismatch_is_rejected(tmp_path: Path) -> None:
    shard = _parquet_shard(
        tmp_path,
        [{"trade_date": "2026-07-21", "ts_code": "000001.SZ", "close": 10.0}],
        logical_name="2026-07-21/a.parquet",
        min_key=["000001.SZ"],
        max_key=["000001.SZ"],
    )
    shard["row_count"] = 2

    with pytest.raises(V17DatasetManifestError, match="row count mismatch"):
        validate_dataset_objects(_parquet_manifest([shard]), repo_root=tmp_path)


def test_parquet_actual_schema_mismatch_is_rejected(tmp_path: Path) -> None:
    pa, pq = _require_pyarrow()
    table = pa.table({"trade_date": ["2026-07-21"], "ts_code": ["000001.SZ"], "close": [10]})
    temporary = tmp_path / "schema-mismatch.parquet"
    pq.write_table(table, temporary)
    raw = temporary.read_bytes()
    digest = _sha256(raw)
    object_path = derive_dataset_object_path(digest, "PARQUET")
    _write_object(tmp_path, object_path, raw)
    shard = {
        "logical_name": "2026-07-21/schema-mismatch.parquet",
        "partition_values": {"trade_date": "2026-07-21"},
        "object_path": object_path,
        "byte_sha256": digest,
        "size_bytes": len(raw),
        "row_count": 1,
        "min_key": ["000001.SZ"],
        "max_key": ["000001.SZ"],
        "schema_sha256": dataset_schema_sha256(PARQUET_SCHEMA),
    }

    with pytest.raises(V17DatasetManifestError, match="logical schema mismatch"):
        validate_dataset_objects(_parquet_manifest([shard]), repo_root=tmp_path)
