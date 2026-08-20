from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import quant_investor.market.pit_universe as pit_module
from quant_investor.market.pit_universe import (
    LIST_STATUS_DELISTED,
    LIST_STATUS_LISTED,
    LIST_STATUS_PENDING,
    PIT_UNIVERSE_EMPTY_PARENT_POINTER,
    PITUniverseRecord,
    PITUniverseStore,
    acquire_pit_universe_capture,
    evaluate_listing_status,
    publish_pit_universe_capture,
    validate_pit_universe_capture,
)


class _CountingProvider:
    def __init__(self, *, extra_listed: bool = False, external_delisted: bool = False) -> None:
        self.calls: list[str] = []
        self.extra_listed = extra_listed
        self.external_delisted = external_delisted

    def stock_basic(
        self,
        *,
        exchange: str,
        list_status: str,
        fields: str,
    ) -> pd.DataFrame:
        assert exchange == ""
        assert fields == ("ts_code,name,area,industry,market,list_date,delist_date,list_status")
        self.calls.append(list_status)
        rows = {
            LIST_STATUS_LISTED: [
                {
                    "ts_code": "000001.SZ",
                    "name": "Active",
                    "area": "SZ",
                    "industry": "Bank",
                    "market": "主板",
                    "list_date": "20200101",
                    "delist_date": "",
                    "list_status": "L",
                }
            ],
            LIST_STATUS_DELISTED: [
                {
                    "ts_code": "000002.SZ",
                    "name": "Delisted",
                    "area": "SZ",
                    "industry": "Tech",
                    "market": "主板",
                    "list_date": "20100101",
                    "delist_date": "20250102",
                    "list_status": "D",
                }
            ],
            LIST_STATUS_PENDING: [],
        }
        if self.extra_listed and list_status == LIST_STATUS_LISTED:
            rows[list_status].append(
                {
                    "ts_code": "000003.SZ",
                    "name": "New",
                    "area": "SZ",
                    "industry": "Tech",
                    "market": "主板",
                    "list_date": "20260819",
                    "delist_date": "",
                    "list_status": "L",
                }
            )
        if self.external_delisted and list_status == LIST_STATUS_DELISTED:
            rows[list_status].append(
                {
                    "ts_code": "T600018.SH",
                    "name": "Legacy Delisted",
                    "area": "SH",
                    "industry": "Transport",
                    "market": None,
                    "list_date": "20000719",
                    "delist_date": "20061020",
                    "list_status": "D",
                }
            )
        return pd.DataFrame(rows[list_status])


def _scope(tmp_path: Path, symbols: list[str]) -> tuple[Path, str]:
    path = tmp_path / "scope.json"
    payload = (json.dumps({"full_a": symbols}, sort_keys=True) + "\n").encode()
    path.write_bytes(payload)
    return path.resolve(), hashlib.sha256(payload).hexdigest()


def _acquire(tmp_path: Path, provider: _CountingProvider) -> dict[str, object]:
    return acquire_pit_universe_capture(
        provider,
        capture_root=(tmp_path / "capture").resolve(),
        observed_at="2026-08-19T09:20:00Z",
        source_run_id="pit-capture-unit",
    )


def test_capture_is_three_calls_and_publish_never_refetches(tmp_path: Path) -> None:
    provider = _CountingProvider()
    capture = _acquire(tmp_path, provider)
    scope_path, scope_sha = _scope(tmp_path, ["000001.SZ"])
    store = PITUniverseStore(root_dir=tmp_path / "reference")

    validation = validate_pit_universe_capture(
        capture["capture_receipt_path"],
        capture["capture_receipt_sha256"],
        store=store,
        canonical_scope_path=scope_path,
        expected_scope_sha256=scope_sha,
    )
    published = publish_pit_universe_capture(
        capture["capture_receipt_path"],
        capture["capture_receipt_sha256"],
        store=store,
        canonical_scope_path=scope_path,
        expected_scope_sha256=scope_sha,
    )

    assert provider.calls == ["L", "D", "P"]
    assert validation["provider_call_count"] == 3
    assert validation["expected_parent_pointer_sha256"] == PIT_UNIVERSE_EMPTY_PARENT_POINTER
    assert validation["provider_accounting"] == {
        "failed": 0,
        "has_more": False,
        "item_count": 2,
        "malformed": 0,
        "canonical_row_count": 2,
        "excluded_provider_external": 0,
        "partition_count": 3,
        "provider_count": 2,
    }
    assert published["execute"] is True
    assert published["compatibility_export_status"] == "written"
    assert published["manifest"]["source_bindings"] == {
        "capture": {
            "schema_version": "cn_pit_universe_capture.v2",
            "path": capture["capture_receipt_path"],
            "sha256": capture["capture_receipt_sha256"],
        },
        "external_exclusion_inventory": capture["exclusion_inventory"],
        "scope_expansion_pending": {
            "schema_version": "cn_pit_scope_expansion_pending.v1",
            "authority_scope": "FROZEN_FULL_A",
            "admission_status": "NOT_CONFIGURED",
            "count": 0,
            "sha256": pit_module._sha256_bytes(pit_module._json_bytes({"items": []})),
            "identities": [],
            "rows": [],
            "transition_count": 0,
            "transition_sha256": pit_module._sha256_bytes(pit_module._json_bytes({"items": []})),
            "transitions": [],
        },
        "full_a_scope": {
            "path": str(scope_path),
            "sha256": scope_sha,
        },
    }
    assert store.load_generation_binding()["generation_id"] == published["generation_id"]
    assert [record.symbol for record in store.load_latest_records()] == [
        "000001.SZ",
        "000002.SZ",
    ]


def test_external_legacy_delisted_identity_is_evidenced_but_never_canonical(
    tmp_path: Path,
) -> None:
    provider = _CountingProvider(external_delisted=True)
    capture = _acquire(tmp_path, provider)
    scope_path, scope_sha = _scope(tmp_path, ["000001.SZ"])
    store = PITUniverseStore(root_dir=tmp_path / "reference")

    validation = validate_pit_universe_capture(
        capture["capture_receipt_path"],
        capture["capture_receipt_sha256"],
        store=store,
        canonical_scope_path=scope_path,
        expected_scope_sha256=scope_sha,
    )
    published = publish_pit_universe_capture(
        capture["capture_receipt_path"],
        capture["capture_receipt_sha256"],
        store=store,
        canonical_scope_path=scope_path,
        expected_scope_sha256=scope_sha,
    )

    assert validation["excluded_provider_external_count"] == 1
    assert validation["provider_accounting"]["canonical_row_count"] == 2
    assert validation["provider_accounting"]["provider_count"] == 3
    inventory = json.loads(Path(capture["exclusion_inventory"]["path"]).read_text())
    assert inventory["items"][0]["identity"] == "T600018.SH"
    assert [record.symbol for record in store.load_latest_records()] == [
        "000001.SZ",
        "000002.SZ",
    ]
    assert "T600018.SH" not in json.dumps(published["manifest"], ensure_ascii=False)


def test_capture_tamper_fails_before_publish(tmp_path: Path) -> None:
    provider = _CountingProvider()
    capture = _acquire(tmp_path, provider)
    scope_path, scope_sha = _scope(tmp_path, ["000001.SZ"])
    partition = Path(capture["partitions"][0]["path"])
    partition.chmod(0o600)
    partition.write_bytes(partition.read_bytes() + b" ")
    partition.chmod(0o400)

    with pytest.raises(RuntimeError, match="partition_sha256_mismatch"):
        publish_pit_universe_capture(
            capture["capture_receipt_path"],
            capture["capture_receipt_sha256"],
            store=PITUniverseStore(root_dir=tmp_path / "reference"),
            canonical_scope_path=scope_path,
            expected_scope_sha256=scope_sha,
        )


def test_new_listed_identity_outside_frozen_scope_is_pending_evidence(tmp_path: Path) -> None:
    provider = _CountingProvider(extra_listed=True)
    capture = _acquire(tmp_path, provider)
    scope_path, scope_sha = _scope(tmp_path, ["000001.SZ"])

    store = PITUniverseStore(root_dir=tmp_path / "reference")
    validation = validate_pit_universe_capture(
        capture["capture_receipt_path"],
        capture["capture_receipt_sha256"],
        store=store,
        canonical_scope_path=scope_path,
        expected_scope_sha256=scope_sha,
    )
    published = publish_pit_universe_capture(
        capture["capture_receipt_path"],
        capture["capture_receipt_sha256"],
        store=store,
        canonical_scope_path=scope_path,
        expected_scope_sha256=scope_sha,
    )
    assert validation["scope_expansion_pending_count"] == 1
    assert validation["dynamic_whole_market_complete"] is False
    assert validation["scope_expansion_pending_rows"][0]["identity"] == "000003.SZ"
    pending = store.records_by_symbol()["000003.SZ"]
    assert pending.membership_quality == "outside_frozen_scope_pending"
    status = evaluate_listing_status(pending, symbol="000003.SZ", as_of="20260819")
    assert status.provider_listed is True
    assert status.authority_membership is False
    assert status.in_universe is False
    assert status.research_eligible is False
    assert status.tradable is False
    assert (
        published["manifest"]["source_bindings"]["scope_expansion_pending"]["admission_status"]
        == "NOT_CONFIGURED"
    )
    assert provider.calls == ["L", "D", "P"]


def test_malformed_provider_identity_is_accounted_and_not_publishable(
    tmp_path: Path,
) -> None:
    class _MalformedProvider(_CountingProvider):
        def stock_basic(self, **kwargs: object) -> pd.DataFrame:
            frame = super().stock_basic(**kwargs)
            if kwargs["list_status"] == LIST_STATUS_LISTED:
                frame.loc[0, "ts_code"] = "bad-symbol"
            return frame

    provider = _MalformedProvider()
    capture = _acquire(tmp_path, provider)
    scope_path, scope_sha = _scope(tmp_path, ["000001.SZ"])

    assert capture["provider_accounting"]["malformed"] == 1
    with pytest.raises(RuntimeError, match="provider_accounting_invalid"):
        validate_pit_universe_capture(
            capture["capture_receipt_path"],
            capture["capture_receipt_sha256"],
            store=PITUniverseStore(root_dir=tmp_path / "reference"),
            canonical_scope_path=scope_path,
            expected_scope_sha256=scope_sha,
        )


def test_expected_parent_is_reread_under_publish_lock(tmp_path: Path) -> None:
    provider = _CountingProvider()
    capture = _acquire(tmp_path, provider)
    scope_path, scope_sha = _scope(tmp_path, ["000001.SZ"])
    store = PITUniverseStore(root_dir=tmp_path / "reference")
    first = store.write_snapshot(
        raw_records=[
            PITUniverseRecord(
                symbol="000001.SZ",
                source_list_status="L",
                list_date="20200101",
                observed_at="2026-08-18T00:00:00Z",
                source_run_id="parent-a",
            )
        ],
        observed_at="2026-08-18T00:00:00Z",
        source_run_id="parent-a",
        source_bindings={"full_a_scope": {"path": str(scope_path), "sha256": scope_sha}},
    )
    expected_parent = first["discovery_pointer_sha256"]
    validate_pit_universe_capture(
        capture["capture_receipt_path"],
        capture["capture_receipt_sha256"],
        store=store,
        canonical_scope_path=scope_path,
        expected_scope_sha256=scope_sha,
        expected_parent_pointer_sha256=expected_parent,
    )
    store.write_snapshot(
        raw_records=[
            PITUniverseRecord(
                symbol="000001.SZ",
                name="Changed",
                source_list_status="L",
                list_date="20200101",
                observed_at="2026-08-18T01:00:00Z",
                source_run_id="parent-b",
            )
        ],
        observed_at="2026-08-18T01:00:00Z",
        source_run_id="parent-b",
        expected_parent_pointer_sha256=expected_parent,
        source_bindings={"full_a_scope": {"path": str(scope_path), "sha256": scope_sha}},
    )

    with pytest.raises(RuntimeError, match="pit_parent_pointer_cas_mismatch"):
        publish_pit_universe_capture(
            capture["capture_receipt_path"],
            capture["capture_receipt_sha256"],
            store=store,
            canonical_scope_path=scope_path,
            expected_scope_sha256=scope_sha,
            expected_parent_pointer_sha256=expected_parent,
        )
    assert provider.calls == ["L", "D", "P"]


def test_compatibility_failure_is_post_cas_retryable_warning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provider = _CountingProvider()
    capture = _acquire(tmp_path, provider)
    scope_path, scope_sha = _scope(tmp_path, ["000001.SZ"])
    store = PITUniverseStore(root_dir=tmp_path / "reference")
    original = pit_module._atomic_write_bytes

    def _fail_compatibility(path: Path, payload: bytes) -> None:
        if path == store.compatibility_path:
            raise OSError("compatibility unavailable")
        original(path, payload)

    monkeypatch.setattr(pit_module, "_atomic_write_bytes", _fail_compatibility)
    published = publish_pit_universe_capture(
        capture["capture_receipt_path"],
        capture["capture_receipt_sha256"],
        store=store,
        canonical_scope_path=scope_path,
        expected_scope_sha256=scope_sha,
    )

    assert published["compatibility_export_status"] == "retryable_warning"
    assert published["warnings"][0].startswith("pit_compatibility_export_failed:")
    assert store.load_generation_binding()["generation_id"] == published["generation_id"]
    assert not store.compatibility_path.exists()


def test_pointer_crash_leaves_orphan_generation_and_same_capture_retries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provider = _CountingProvider()
    capture = _acquire(tmp_path, provider)
    scope_path, scope_sha = _scope(tmp_path, ["000001.SZ"])
    store = PITUniverseStore(root_dir=tmp_path / "reference")
    original = pit_module._atomic_write_bytes

    def _crash_before_pointer(path: Path, payload: bytes) -> None:
        if path == store.manifest_path:
            raise OSError("simulated pointer crash")
        original(path, payload)

    monkeypatch.setattr(pit_module, "_atomic_write_bytes", _crash_before_pointer)
    with pytest.raises(OSError, match="simulated pointer crash"):
        publish_pit_universe_capture(
            capture["capture_receipt_path"],
            capture["capture_receipt_sha256"],
            store=store,
            canonical_scope_path=scope_path,
            expected_scope_sha256=scope_sha,
        )
    assert not store.manifest_path.exists()
    assert len(list(store.generations_root.iterdir())) == 1

    monkeypatch.setattr(pit_module, "_atomic_write_bytes", original)
    retried = publish_pit_universe_capture(
        capture["capture_receipt_path"],
        capture["capture_receipt_sha256"],
        store=store,
        canonical_scope_path=scope_path,
        expected_scope_sha256=scope_sha,
    )
    assert store.load_generation_binding()["generation_id"] == retried["generation_id"]
    assert len(list(store.generations_root.iterdir())) == 1
    assert provider.calls == ["L", "D", "P"]


def test_concurrent_bootstrap_rejects_empty_parent_and_retains_orphan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provider = _CountingProvider()
    capture = _acquire(tmp_path, provider)
    scope_path, scope_sha = _scope(tmp_path, ["000001.SZ"])
    store = PITUniverseStore(root_dir=tmp_path / "reference")
    original_atomic = pit_module._atomic_write_bytes

    def _crash_before_first_pointer(path: Path, payload: bytes) -> None:
        if path == store.manifest_path:
            raise OSError("simulated bootstrap crash")
        original_atomic(path, payload)

    monkeypatch.setattr(
        pit_module,
        "_atomic_write_bytes",
        _crash_before_first_pointer,
    )
    with pytest.raises(OSError, match="simulated bootstrap crash"):
        publish_pit_universe_capture(
            capture["capture_receipt_path"],
            capture["capture_receipt_sha256"],
            store=store,
            canonical_scope_path=scope_path,
            expected_scope_sha256=scope_sha,
        )
    orphan_generation_names = {path.name for path in store.generations_root.iterdir()}
    assert len(orphan_generation_names) == 1
    assert not store.manifest_path.exists()

    monkeypatch.setattr(pit_module, "_atomic_write_bytes", original_atomic)
    original_write_snapshot = store.write_snapshot
    winner: dict[str, object] = {}

    def _race_after_validation(**kwargs: object) -> dict[str, object]:
        winner.update(
            original_write_snapshot(
                raw_records=[
                    PITUniverseRecord(
                        symbol="000001.SZ",
                        source_list_status="L",
                        list_date="20200101",
                        observed_at="2026-08-19T09:21:00Z",
                        source_run_id="concurrent-bootstrap-winner",
                    )
                ],
                observed_at="2026-08-19T09:21:00Z",
                source_run_id="concurrent-bootstrap-winner",
            )
        )
        return original_write_snapshot(**kwargs)

    monkeypatch.setattr(store, "write_snapshot", _race_after_validation)

    with pytest.raises(RuntimeError, match="pit_parent_pointer_cas_mismatch"):
        publish_pit_universe_capture(
            capture["capture_receipt_path"],
            capture["capture_receipt_sha256"],
            store=store,
            canonical_scope_path=scope_path,
            expected_scope_sha256=scope_sha,
        )
    assert store.load_generation_binding()["generation_id"] == winner["generation_id"]
    final_generation_names = {path.name for path in store.generations_root.iterdir()}
    assert orphan_generation_names <= final_generation_names
    assert len(final_generation_names) == 2


def test_shadow_candidate_is_valid_private_store_without_production_pointer(
    tmp_path: Path,
) -> None:
    provider = _CountingProvider()
    capture = _acquire(tmp_path, provider)
    scope_path, scope_sha = _scope(tmp_path, ["000001.SZ"])
    store = PITUniverseStore(root_dir=tmp_path / "reference")

    shadow = publish_pit_universe_capture(
        capture["capture_receipt_path"],
        capture["capture_receipt_sha256"],
        store=store,
        canonical_scope_path=scope_path,
        expected_scope_sha256=scope_sha,
        canonical=False,
        shadow_root=(tmp_path / "shadow").resolve(),
    )

    assert shadow["execute"] is False
    assert shadow["shadow_candidate"]["canonical_write_authorized"] is False
    private_root = (tmp_path / "shadow" / "reference").resolve()
    private_store = PITUniverseStore(root_dir=private_root)
    binding = private_store.load_generation_binding()
    assert shadow["generation_id"] == binding["generation_id"]
    assert shadow["generation_manifest_path"] == binding["generation_manifest_path"]
    assert shadow["generation_manifest_sha256"] == binding["generation_manifest_sha256"]
    assert shadow["canonical_path"] == binding["canonical_path"]
    assert shadow["canonical_sha256"] == binding["canonical_sha256"]
    assert shadow["discovery_pointer_path"] == binding["discovery_pointer_path"]
    assert shadow["discovery_pointer_sha256"] == binding["discovery_pointer_sha256"]
    assert Path(shadow["canonical_path"]).is_relative_to(
        tmp_path / "shadow" / "reference" / "_generations"
    )
    assert shadow["compatibility_export_status"] == "skipped"
    assert not (tmp_path / "shadow" / "compatibility-export-disabled.json").exists()
    assert not store.manifest_path.exists()
    assert not store.generations_root.exists()
