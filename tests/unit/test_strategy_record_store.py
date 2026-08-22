from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from quant_investor.strategy_records import store as store_module

from quant_investor.strategy_records.store import (
    ARCHIVE_LOCATOR_SCHEMA,
    ARCHIVE_MANIFEST_SCHEMA,
    ARCHIVE_RESTORE_RECEIPT_SCHEMA,
    CATALOG_SCHEMA_V2,
    EMPTY_POINTER_SHA256,
    StrategyRecordCASMismatch,
    StrategyRecordConflict,
    StrategyRecordStoreError,
    bootstrap_catalog,
    canonical_json_bytes,
    catalog_history_entries,
    content_sha256,
    catalog_online_record_dirs,
    load_registered_catalog,
    publish_catalog,
    reselect_catalog,
    resolve_active_record_dirs,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(record_id: str) -> dict[str, object]:
    return {
        "record_id": record_id,
        "relative_path": record_id,
        "state": "ONLINE",
        "storage_state": "ONLINE",
        "sealed_at": "2026-08-10T00:00:00Z",
        "inventory": [],
        "inventory_sha256": hashlib.sha256(b"[]\n").hexdigest(),
        "file_count": 0,
        "total_bytes": 0,
    }


def _bootstrap(root: Path) -> dict[str, object]:
    (root / "r1").mkdir()
    (root / "r2").mkdir()
    return bootstrap_catalog(
        root,
        records=[_record("r1"), _record("r2")],
        active_record_id="r2",
        previous_record_id="r1",
        generation_id="g1",
        published_at="2026-08-10T00:00:00Z",
    )


def test_unregistered_legacy_root_is_not_implicitly_authoritative(
    tmp_path: Path,
) -> None:
    (tmp_path / "legacy-run").mkdir()
    assert load_registered_catalog(tmp_path) is None


def test_bootstrap_registers_single_pointer_and_resolvers(
    tmp_path: Path,
) -> None:
    result = _bootstrap(tmp_path)
    loaded = load_registered_catalog(tmp_path)
    assert loaded == (result["pointer"], result["catalog"])
    assert [path.name for path in catalog_online_record_dirs(tmp_path)] == [
        "r1",
        "r2",
    ]
    assert [path.name for path in resolve_active_record_dirs(tmp_path)] == [
        "r2",
        "r1",
    ]
    history = catalog_history_entries(tmp_path)
    assert [row["storage_state"] for row in history] == ["ONLINE", "ONLINE"]
    assert all(row["record_dir"] for row in history)


def test_registered_state_is_fail_closed_on_missing_pointer(
    tmp_path: Path,
) -> None:
    (tmp_path / "_record_store").mkdir()
    with pytest.raises(StrategyRecordStoreError, match="no current pointer"):
        load_registered_catalog(tmp_path)


def test_catalog_corruption_and_pointer_hardlink_are_rejected(
    tmp_path: Path,
) -> None:
    result = _bootstrap(tmp_path)
    catalog_path = tmp_path / result["pointer"]["catalog_path"]
    original = catalog_path.read_bytes()
    catalog_path.write_bytes(original.replace(b'"record_count":2', b'"record_count":9'))
    with pytest.raises(StrategyRecordStoreError):
        load_registered_catalog(tmp_path)

    catalog_path.write_bytes(original)
    pointer = tmp_path / "_record_store" / "current.v1.json"
    os.link(pointer, tmp_path / "pointer-hardlink.json")
    with pytest.raises(StrategyRecordStoreError, match="single-link"):
        load_registered_catalog(tmp_path)


def test_publish_requires_pointer_cas_and_keeps_pointer_on_failure(
    tmp_path: Path,
) -> None:
    _bootstrap(tmp_path)
    pointer_path = tmp_path / "_record_store" / "current.v1.json"
    before = pointer_path.read_bytes()
    with pytest.raises(StrategyRecordCASMismatch):
        publish_catalog(
            tmp_path,
            expected_pointer_sha256="0" * 64,
            generation_id="g2",
            published_at="2026-08-10T00:01:00Z",
        )
    assert pointer_path.read_bytes() == before


def test_immutable_catalog_identity_is_idempotent_or_conflicts(
    tmp_path: Path,
) -> None:
    result = _bootstrap(tmp_path)
    pointer_sha = _sha(tmp_path / "_record_store" / "current.v1.json")
    first = publish_catalog(
        tmp_path,
        expected_pointer_sha256=pointer_sha,
        records=result["catalog"]["records"],
        active_record_id="r2",
        previous_record_id="r1",
        generation_id="g2",
        published_at="2026-08-10T00:01:00Z",
    )
    assert first["pointer"]["generation_id"] == "g2"
    with pytest.raises(StrategyRecordConflict, match="collision"):
        publish_catalog(
            tmp_path,
            expected_pointer_sha256=first["pointer_sha256"],
            generation_id="g2",
            published_at="2026-08-10T00:02:00Z",
        )


def test_reselect_catalog_uses_exact_existing_generation_and_pointer_cas(
    tmp_path: Path,
) -> None:
    (tmp_path / "r1").mkdir()
    (tmp_path / "r2").mkdir()
    first = bootstrap_catalog(
        tmp_path,
        records=[_record("r1"), _record("r2")],
        active_record_id="r2",
        previous_record_id="r1",
        generation_id="g1",
        published_at="2026-08-10T00:00:00Z",
        catalog_schema=CATALOG_SCHEMA_V2,
    )
    original_catalog = (tmp_path / first["pointer"]["catalog_path"]).read_bytes()
    second = publish_catalog(
        tmp_path,
        expected_pointer_sha256=first["pointer_sha256"],
        generation_id="g2",
        published_at="2026-08-10T00:01:00Z",
        catalog_schema=CATALOG_SCHEMA_V2,
    )
    result = reselect_catalog(
        tmp_path,
        expected_current_pointer_sha256=second["pointer_sha256"],
        target_generation_id=first["pointer"]["generation_id"],
        target_catalog_path=first["pointer"]["catalog_path"],
        target_catalog_sha256=first["pointer"]["catalog_sha256"],
        published_at="2026-08-10T00:02:00Z",
    )
    assert result["catalog_reselected"] is True
    assert result["catalog_created"] is False
    assert result["pointer"]["generation_id"] == "g1"
    assert result["pointer"]["previous_pointer_sha256"] == second["pointer_sha256"]
    assert result["catalog"] == first["catalog"]
    assert (tmp_path / first["pointer"]["catalog_path"]).read_bytes() == original_catalog
    assert load_registered_catalog(tmp_path) == (
        result["pointer"],
        first["catalog"],
    )


def test_reselect_catalog_rejects_cas_or_target_sha_without_pointer_change(
    tmp_path: Path,
) -> None:
    (tmp_path / "r1").mkdir()
    (tmp_path / "r2").mkdir()
    first = bootstrap_catalog(
        tmp_path,
        records=[_record("r1"), _record("r2")],
        active_record_id="r2",
        previous_record_id="r1",
        generation_id="g1",
        published_at="2026-08-10T00:00:00Z",
        catalog_schema=CATALOG_SCHEMA_V2,
    )
    second = publish_catalog(
        tmp_path,
        expected_pointer_sha256=first["pointer_sha256"],
        generation_id="g2",
        published_at="2026-08-10T00:01:00Z",
        catalog_schema=CATALOG_SCHEMA_V2,
    )
    pointer_path = tmp_path / "_record_store/current.v1.json"
    before = pointer_path.read_bytes()
    values = {
        "target_generation_id": "g1",
        "target_catalog_path": first["pointer"]["catalog_path"],
        "target_catalog_sha256": first["pointer"]["catalog_sha256"],
        "published_at": "2026-08-10T00:02:00Z",
    }
    with pytest.raises(StrategyRecordCASMismatch):
        reselect_catalog(
            tmp_path,
            expected_current_pointer_sha256="0" * 64,
            **values,
        )
    assert pointer_path.read_bytes() == before
    values["target_catalog_sha256"] = "0" * 64
    with pytest.raises(StrategyRecordStoreError, match="catalog byte SHA-256"):
        reselect_catalog(
            tmp_path,
            expected_current_pointer_sha256=second["pointer_sha256"],
            **values,
        )
    assert pointer_path.read_bytes() == before


def test_pointer_catalog_sha_is_exact_bytes_not_json_semantics(
    tmp_path: Path,
) -> None:
    result = _bootstrap(tmp_path)
    pointer_path = tmp_path / "_record_store" / "current.v1.json"
    pointer = json.loads(pointer_path.read_text())
    catalog_path = tmp_path / pointer["catalog_path"]
    assert pointer["catalog_sha256"] == _sha(catalog_path)
    assert result["pointer_sha256"] == _sha(pointer_path)
    assert EMPTY_POINTER_SHA256 == hashlib.sha256(b"").hexdigest()


def _write_sealed(path: Path, value: dict[str, object]) -> str:
    body = dict(value)
    body["content_sha256"] = content_sha256(body)
    raw = canonical_json_bytes(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_catalog_v2_validates_archive_closure_and_keeps_active_online(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    root = project / "results/strategy_records/CN/aggressive_tech_manufacturing"
    for record_id in ("20260601_1000", "20260809_1000", "20260810_1000"):
        (root / record_id).mkdir(parents=True)
    initial = bootstrap_catalog(
        root,
        records=[
            _record("20260601_1000"),
            _record("20260809_1000"),
            _record("20260810_1000"),
        ],
        active_record_id="20260810_1000",
        previous_record_id="20260809_1000",
        generation_id="g1",
        published_at="2026-08-10T00:00:00Z",
    )
    month = project / (
        "results/strategy_record_archives/CN/aggressive_tech_manufacturing/" "monthly/v1/2026-06"
    )
    archive = month / "archive.tar.zst"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"archive")
    archive_sha = _sha(archive)
    archive_relative = archive.relative_to(project).as_posix()
    archived = _record("20260601_1000")
    manifest_record = {
        key: archived[key]
        for key in (
            "record_id",
            "relative_path",
            "inventory",
            "inventory_sha256",
            "file_count",
            "total_bytes",
        )
    }
    manifest_record["member_prefix"] = archived["relative_path"]
    manifest_path = month / "archive-manifest.v1.json"
    manifest_relative = manifest_path.relative_to(project).as_posix()
    manifest_sha = _write_sealed(
        manifest_path,
        {
            "schema_id": ARCHIVE_MANIFEST_SCHEMA,
            "archive_id": "archive-2026-06",
            "archive_path": archive_relative,
            "archive_sha256": archive_sha,
            "archive_bytes": len(b"archive"),
            "records": [manifest_record],
            "record_count": 1,
            "file_count": 0,
            "logical_bytes": 0,
        },
    )
    receipt_path = month / "restore-receipt.v1.json"
    receipt_relative = receipt_path.relative_to(project).as_posix()
    receipt_sha = _write_sealed(
        receipt_path,
        {
            "schema_id": ARCHIVE_RESTORE_RECEIPT_SCHEMA,
            "archive_id": "archive-2026-06",
            "archive_path": archive_relative,
            "archive_sha256": archive_sha,
            "manifest_path": manifest_relative,
            "manifest_sha256": manifest_sha,
            "record_ids": ["20260601_1000"],
            "record_count": 1,
            "restored_file_count": 0,
            "restored_logical_bytes": 0,
            "all_inventory_matched": True,
        },
    )
    records = [dict(row) for row in initial["catalog"]["records"]]
    records[0].update(
        {
            "state": "ARCHIVED",
            "storage_state": "ARCHIVED",
            "archive_locator": {
                "schema_id": ARCHIVE_LOCATOR_SCHEMA,
                "archive_id": "archive-2026-06",
                "archive_path": archive_relative,
                "archive_sha256": archive_sha,
                "archive_bytes": len(b"archive"),
                "manifest_path": manifest_relative,
                "manifest_sha256": manifest_sha,
                "restore_receipt_path": receipt_relative,
                "restore_receipt_sha256": receipt_sha,
                "member_prefix": "20260601_1000",
            },
        }
    )
    published = publish_catalog(
        root,
        expected_pointer_sha256=initial["pointer_sha256"],
        records=records,
        active_record_id="20260810_1000",
        previous_record_id="20260809_1000",
        generation_id="g2",
        published_at="2026-08-10T00:01:00Z",
        catalog_schema=CATALOG_SCHEMA_V2,
    )
    assert published["catalog"]["schema_id"] == CATALOG_SCHEMA_V2
    assert catalog_history_entries(root)[0]["record_dir"] is None

    bad = [dict(row) for row in records]
    bad[1]["state"] = "ARCHIVED"
    bad[1]["storage_state"] = "ARCHIVED"
    bad[1]["archive_locator"] = records[0]["archive_locator"]
    with pytest.raises(StrategyRecordStoreError, match="active record is not ONLINE"):
        publish_catalog(
            root,
            expected_pointer_sha256=published["pointer_sha256"],
            records=bad,
            active_record_id="20260809_1000",
            previous_record_id="20260810_1000",
            generation_id="g3",
            catalog_schema=CATALOG_SCHEMA_V2,
        )


def test_catalog_v2_rejects_state_pair_mismatch(tmp_path: Path) -> None:
    (tmp_path / "r1").mkdir()
    (tmp_path / "r2").mkdir()
    records = [_record("r1"), _record("r2")]
    records[0]["storage_state"] = "ARCHIVED"
    with pytest.raises(StrategyRecordStoreError, match="state/storage_state"):
        bootstrap_catalog(
            tmp_path,
            records=records,
            active_record_id="r2",
            previous_record_id="r1",
            generation_id="g1",
            catalog_schema=CATALOG_SCHEMA_V2,
        )


def _late_delay_record() -> dict:
    return {
        "record_id": "20260822_0930",
        "sealed_at": "2026-08-22T01:31:00Z",
        "publication_delay": {
            "schema_id": store_module.PUBLICATION_DELAY_SCHEMA,
            "publication_class": store_module.LATE_OFFICIAL_VALUATION_PUBLICATION,
            "expected_valuation_date": "2026-08-21",
            "expected_publication_date": "2026-08-22",
            "publication_delay_reason": store_module.LATE_PUBLICATION_REASON,
            "actual_sealed_at": "2026-08-22T01:31:00Z",
            "actual_published_at": "2026-08-22T01:31:00Z",
            "actual_publication_local_date": "2026-08-22",
            "candidate_recorded_at": "2026-08-22T09:30:00+08:00",
            "continuity_receipt_id": "automation-20260821-daily-review-v1",
            "continuity_receipt_sha256": "a" * 64,
            "continuity_receipt_created_at": "2026-08-21T13:27:37Z",
            "continuity_checkpoint_digest": "b" * 64,
            "source_record": "20260820_1321",
            "evidence_date": "2026-08-21",
            "delay_days": 1,
            "historical_holdings_storage_authority": True,
            "v17_mainline_authority": False,
            "broker_order_trade_authority": False,
        },
    }


def test_publication_delay_metadata_is_typed_and_time_ordered() -> None:
    record = _late_delay_record()
    store_module._validate_publication_delay(record)
    record["publication_delay"]["delay_days"] = 2
    with pytest.raises(StrategyRecordStoreError, match="contract is invalid"):
        store_module._validate_publication_delay(record)


def test_publication_delay_metadata_rejects_recorded_minute_drift() -> None:
    record = _late_delay_record()
    record["publication_delay"]["candidate_recorded_at"] = "2026-08-22T09:31:00+08:00"
    with pytest.raises(StrategyRecordStoreError, match="record minute mismatch"):
        store_module._validate_publication_delay(record)
