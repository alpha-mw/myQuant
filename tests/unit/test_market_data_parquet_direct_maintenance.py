from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import quant_investor.market.download as download_module
import quant_investor.market.download_cn as download_cn_module
from quant_investor.market.cn_nontrading_evidence import (
    build_bak_daily_nontrading_evidence,
    canonical_json_sha256,
    evidence_cache_path,
    file_sha256,
    symbol_set_sha256,
    write_evidence_cache,
)
from quant_investor.market.market_data_store import MarketDataStore
from quant_investor.market.market_data_reader import coverage_fingerprint
from quant_investor.market.pit_universe import PITUniverseRecord, PITUniverseStore
from tests.fixtures.strict_cn_snapshot import (
    coverage_v4 as strict_coverage_v4,
    v4_snapshot_paths,
)


def test_cn_legacy_maintenance_is_disabled_outside_staged():
    with pytest.raises(ValueError, match="CN legacy CSV maintenance is disabled"):
        download_module.run_market_maintenance(
            market="CN",
            categories=["full_a"],
            storage_mode="legacy",
        )


def _write_pit_generation(
    root: Path,
    *,
    records: list[PITUniverseRecord] | None = None,
) -> dict[str, object]:
    selected_records = records or [
        PITUniverseRecord(
            symbol="000001.SZ",
            name="One",
            list_date="20200101",
            source_list_status="L",
            observed_at="2026-03-16T00:00:00Z",
            source_run_id="unit-pit",
        ),
        PITUniverseRecord(
            symbol="000002.SZ",
            name="Two",
            list_date="20200101",
            source_list_status="L",
            observed_at="2026-03-16T00:00:00Z",
            source_run_id="unit-pit",
        ),
    ]
    store = PITUniverseStore(
        root_dir=root / "parquet" / "cn" / "reference",
        raw_root=root / "cn_universe" / "raw",
        compatibility_path=(root / "cn_universe" / "stock_basic_membership_latest.json"),
    )
    return store.write_snapshot(
        raw_records=selected_records,
        latest_records=selected_records,
        observed_at="2026-03-16T00:00:00Z",
        source_run_id="unit-pit",
    )


def _production_binding_args(
    root: Path,
    generation: dict[str, object],
) -> dict[str, str]:
    return {
        "pit_generation_manifest": str(generation["generation_manifest_path"]),
        "expected_pit_generation_manifest_sha256": str(generation["generation_manifest_sha256"]),
        "expected_market_pointer_sha256": file_sha256(root / "parquet" / "cn" / "_latest.json"),
    }


def test_cn_auto_maintenance_uses_parquet_direct(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class FakeMaintainer:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def maintain(self, **kwargs):
            captured["maintain"] = kwargs
            return {"storage_mode": "parquet-direct"}

    monkeypatch.setattr(download_module, "CNParquetBatchMaintainer", FakeMaintainer)
    _write_seed_snapshot(tmp_path)
    generation = _write_pit_generation(tmp_path)
    monkeypatch.setattr(
        download_module.config,
        "MARKET_DATA_BASE_DIR",
        str(tmp_path),
        raising=False,
    )

    result = download_module.run_market_maintenance(
        market="CN",
        categories=["full_a"],
        target_date="20260316",
        **_production_binding_args(tmp_path, generation),
    )

    assert result["storage_mode"] == "parquet-direct"
    assert captured["maintain"]["categories"] == ["full_a"]
    assert captured["maintain"]["target_date"] == "20260316"
    assert (
        captured["maintain"]["pit_generation_binding"]["generation_id"]
        == generation["generation_id"]
    )


def test_cn_parquet_direct_requires_publish_bindings_before_provider_init(
    monkeypatch,
):
    def _unexpected_init(**_kwargs):
        raise AssertionError("provider-backed maintainer must not initialize")

    monkeypatch.setattr(
        download_module,
        "CNParquetBatchMaintainer",
        _unexpected_init,
    )
    with pytest.raises(
        ValueError,
        match="expected_market_pointer_sha256_invalid",
    ):
        download_module.run_market_maintenance(
            market="CN",
            categories=["full_a"],
            storage_mode="parquet-direct",
        )


def _write_seed_snapshot(root: Path) -> None:
    table_root, serving_root, manifest_path = v4_snapshot_paths(root, "seed")
    month_dir = table_root / "year=2026" / "month=03"
    symbol_1 = serving_root / "symbol=000001.SZ"
    symbol_2 = serving_root / "symbol=000002.SZ"
    month_dir.mkdir(parents=True)
    symbol_1.mkdir(parents=True)
    symbol_2.mkdir(parents=True)
    frame = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20260315",
                "open": 9.5,
                "high": 10.0,
                "low": 9.0,
                "close": 9.8,
                "vol": 900,
                "amount": 9000.0,
                "adj_factor": 1.0,
            },
            {
                "ts_code": "000002.SZ",
                "trade_date": "20260315",
                "open": 19.5,
                "high": 20.0,
                "low": 19.0,
                "close": 19.8,
                "vol": 1900,
                "amount": 19000.0,
                "adj_factor": 1.0,
            },
        ]
    )
    frame.to_parquet(month_dir / "part.parquet", index=False)
    legacy_month_dir = root / "parquet" / "cn" / "bars" / "year=2026" / "month=03"
    legacy_month_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(legacy_month_dir / "part.parquet", index=False)
    legacy_serving_root = root / "parquet_serving" / "cn" / "bars"
    (legacy_serving_root / "symbol=000001.SZ").mkdir(parents=True, exist_ok=True)
    (legacy_serving_root / "symbol=000002.SZ").mkdir(parents=True, exist_ok=True)
    frame[frame["ts_code"].eq("000001.SZ")].to_parquet(
        legacy_serving_root / "symbol=000001.SZ" / "bars.parquet",
        index=False,
    )
    frame[frame["ts_code"].eq("000002.SZ")].to_parquet(
        legacy_serving_root / "symbol=000002.SZ" / "bars.parquet",
        index=False,
    )
    frame[frame["ts_code"].eq("000001.SZ")].to_parquet(symbol_1 / "bars.parquet", index=False)
    frame[frame["ts_code"].eq("000002.SZ")].to_parquet(symbol_2 / "bars.parquet", index=False)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    coverage = strict_coverage_v4(
        root,
        ["000001.SZ", "000002.SZ"],
        trade_date="20260315",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "snapshot_id": "seed",
                "status": "OK",
                "table_root": str(table_root),
                "derived_serving_root": str(serving_root),
                "latest_complete_trade_date": "20260315",
                "coverage": coverage,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (root / "parquet" / "cn" / "_latest.json").write_text(
        json.dumps(
            {
                "snapshot_id": "seed",
                "status": "OK",
                "manifest_path": str(manifest_path),
                "table_root": str(table_root),
                "derived_serving_root": str(serving_root),
                "latest_available_trade_date": "20260315",
                "latest_complete_trade_date": "20260315",
                "latest_trade_date": "20260315",
                "coverage": coverage,
                "blockers": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_upsert_bars_merges_target_day_without_replacing_history(tmp_path):
    _write_seed_snapshot(tmp_path)
    store = MarketDataStore(market="CN", data_root=tmp_path)
    latest_path = tmp_path / "parquet" / "cn" / "_latest.json"
    previous_pointer_sha256 = hashlib.sha256(latest_path.read_bytes()).hexdigest()
    legacy_table_path = (
        tmp_path / "parquet" / "cn" / "bars" / "year=2026" / "month=03" / "part.parquet"
    )
    legacy_table_bytes = legacy_table_path.read_bytes()

    manifest = store.upsert_bars(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20260316",
                    "open": 10.0,
                    "high": 10.6,
                    "low": 9.9,
                    "close": 10.4,
                    "vol": 1000,
                    "amount": 12000.0,
                    "adj_factor": 1.1,
                },
                {
                    "ts_code": "000003.SZ",
                    "trade_date": "20260316",
                    "open": 30.0,
                    "high": 30.8,
                    "low": 29.8,
                    "close": 30.4,
                    "vol": 3000,
                    "amount": 32000.0,
                    "adj_factor": 1.0,
                },
            ]
        ),
        target_trade_date="20260316",
        source="unit-test",
        snapshot_id="upserted",
        expected_latest_pointer_sha256=previous_pointer_sha256,
        metadata={
            "status": "OK",
            "latest_available_trade_date": "20260316",
            "latest_complete_trade_date": "20260316",
            "coverage": strict_coverage_v4(
                tmp_path,
                ["000001.SZ", "000002.SZ", "000003.SZ"],
                trade_date="20260316",
            ),
        },
    )

    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    table_root = Path(latest["table_root"])
    serving_root = Path(latest["derived_serving_root"])
    table = pd.read_parquet(table_root / "year=2026" / "month=03" / "part.parquet")
    assert manifest["snapshot_id"] == "upserted"
    assert manifest["row_count"] == 4
    assert table_root == (
        tmp_path / "parquet" / "cn" / "_snapshots" / "upserted" / "table" / "bars"
    )
    assert serving_root == (
        tmp_path / "parquet" / "cn" / "_snapshots" / "upserted" / "serving" / "bars"
    )
    assert legacy_table_path.read_bytes() == legacy_table_bytes
    assert set(table["ts_code"]) == {"000001.SZ", "000002.SZ", "000003.SZ"}
    assert set(table["trade_date"]) == {"20260315", "20260316"}
    assert latest["snapshot_id"] == "upserted"
    assert store.validate_latest()["status"] == "passed"
    assert store.reader.snapshot()["coverage"]["coverage_trade_date"] == "20260316"


def test_historical_multi_date_upsert_preserves_latest_coverage(tmp_path):
    _write_seed_snapshot(tmp_path)
    latest_path = tmp_path / "parquet" / "cn" / "_latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    coverage = dict(latest["coverage"])
    latest["coverage"] = coverage
    manifest_path = Path(latest["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["coverage"] = coverage
    latest_path.write_text(json.dumps(latest), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    before_fingerprint = coverage_fingerprint(coverage)
    store = MarketDataStore(market="CN", data_root=tmp_path)

    incoming = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": trade_date,
                "open": 10.0,
                "high": 10.5,
                "low": 9.5,
                "close": 10.2,
                "vol": 1000.0,
                "amount": 10000.0,
                "adj_factor": 1.0,
            }
            for trade_date in ["20260313", "20260314"]
        ]
    )
    repaired = store.upsert_bars(
        incoming,
        target_trade_date="20260314",
        target_trade_dates=["20260313", "20260314"],
        source="unit-history-repair",
        metadata={
            "latest_available_trade_date": "20260315",
            "latest_complete_trade_date": "20260315",
            "coverage": {"coverage_trade_date": "20260314"},
        },
    )

    after = json.loads(latest_path.read_text(encoding="utf-8"))
    assert repaired["historical_upsert_coverage_preserved"] is True
    assert after["latest_complete_trade_date"] == "20260315"
    assert coverage_fingerprint(after["coverage"]) == before_fingerprint
    table = pd.read_parquet(Path(after["table_root"]) / "year=2026" / "month=03" / "part.parquet")
    assert {"20260313", "20260314"}.issubset(set(table["trade_date"]))


def test_upsert_rejects_market_pointer_cas_drift_without_publishing(tmp_path):
    _write_seed_snapshot(tmp_path)
    latest_path = tmp_path / "parquet" / "cn" / "_latest.json"
    before = latest_path.read_bytes()
    store = MarketDataStore(market="CN", data_root=tmp_path)

    with pytest.raises(ValueError, match="market_pointer_cas_mismatch"):
        store.upsert_bars(
            pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "20260316",
                        "open": 10.0,
                        "high": 10.6,
                        "low": 9.9,
                        "close": 10.4,
                        "vol": 1000,
                        "amount": 12000.0,
                        "adj_factor": 1.1,
                    }
                ]
            ),
            target_trade_date="20260316",
            source="unit-test-cas",
            snapshot_id="cas-rejected",
            expected_latest_pointer_sha256="0" * 64,
        )

    assert latest_path.read_bytes() == before
    assert not (tmp_path / "parquet" / "cn" / "_snapshots" / "cas-rejected").exists()


@pytest.mark.parametrize("snapshot_id", ["..", "../escape", "nested/name"])
def test_upsert_rejects_unsafe_snapshot_id(tmp_path, snapshot_id):
    _write_seed_snapshot(tmp_path)
    latest_path = tmp_path / "parquet" / "cn" / "_latest.json"
    before = latest_path.read_bytes()
    store = MarketDataStore(market="CN", data_root=tmp_path)

    with pytest.raises(ValueError, match="snapshot_id_invalid"):
        store.upsert_bars(
            pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "20260316",
                        "open": 10.0,
                        "high": 10.6,
                        "low": 9.9,
                        "close": 10.4,
                        "vol": 1000,
                        "amount": 12000.0,
                        "adj_factor": 1.1,
                    }
                ]
            ),
            target_trade_date="20260316",
            source="unit-test-invalid-snapshot-id",
            snapshot_id=snapshot_id,
        )

    assert latest_path.read_bytes() == before


def test_storage_validate_fails_closed_for_unbound_complete_coverage(tmp_path):
    _write_seed_snapshot(tmp_path)
    latest_path = tmp_path / "parquet" / "cn" / "_latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    latest["coverage"] = {
        "complete": True,
        "blocking_incomplete_count": 0,
        "expected_scope_count": 2,
    }
    latest_path.write_text(json.dumps(latest), encoding="utf-8")

    validation = MarketDataStore(market="CN", data_root=tmp_path).validate_latest()

    assert validation["status"] == "failed"
    assert "coverage_trade_date_missing" in validation["blockers"]
    assert "coverage_expected_scope_sha256_missing" in validation["blockers"]
    assert "coverage_complete_count_missing_or_invalid" in validation["blockers"]
    assert "coverage_ratio_not_one" in validation["blockers"]


def test_storage_validate_rejects_historical_scope_hash_backfill_provenance(tmp_path):
    _write_seed_snapshot(tmp_path)
    latest_path = tmp_path / "parquet" / "cn" / "_latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    latest["coverage"] = {
        "complete": True,
        "blocking_incomplete_count": 0,
        "expected_scope_count": 2,
        "coverage_trade_date": "20260315",
        "expected_scope_sha256": "a" * 64,
    }
    latest_path.write_text(json.dumps(latest), encoding="utf-8")
    manifest_path = Path(latest["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["coverage"] = latest["coverage"]
    manifest["historical_scope_hash_backfilled"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    store = MarketDataStore(market="CN", data_root=tmp_path)
    validation = store.validate_latest()

    assert validation["status"] == "failed"
    assert "coverage_scope_hash_backfilled_from_historical_target" in (validation["blockers"])
    assert store.reader.snapshot()["coverage_provenance_blockers"] == [
        "coverage_scope_hash_backfilled_from_historical_target"
    ]


def test_storage_validate_semantically_reads_v3_nontrading_evidence(tmp_path):
    _write_seed_snapshot(tmp_path)
    coverage = strict_coverage_v4(
        tmp_path,
        ["000001.SZ", "000002.SZ", "000003.SZ"],
        trade_date="20260315",
        observed_bar_count=2,
    )
    pit_path = Path(coverage["pit_membership_path"])
    pit_sha = str(coverage["pit_membership_sha256"])
    evidence_path = tmp_path / "cn_market_full" / ".cache" / "evidence.json"
    payload = build_bak_daily_nontrading_evidence(
        pd.DataFrame(
            [
                {
                    "ts_code": "000003.SZ",
                    "trade_date": "20260315",
                    "open": 0.0,
                    "high": 0.0,
                    "low": 0.0,
                    "close": 10.0,
                    "pre_close": 10.0,
                    "change": 0.0,
                    "pct_chg": 0.0,
                    "vol": 0.0,
                    "amount": 0.0,
                }
            ]
        ),
        trade_date="20260315",
        primary_missing_symbols=["000003.SZ"],
        query_params={"trade_date": "20260315"},
        pit_membership_path=pit_path,
        pit_membership_sha256=pit_sha,
    )
    write_evidence_cache(evidence_path, payload)
    coverage.update(
        {
            "coverage_ratio": 1.0,
            "coverage_complete_count": 3,
            "verified_nontrading_bak_daily_zero_symbols": ["000003.SZ"],
            "non_blocking_absent_symbols": ["000003.SZ"],
            "verified_nontrading_evidence_path": str(evidence_path),
            "verified_nontrading_evidence_sha256": file_sha256(evidence_path),
        }
    )
    latest_path = tmp_path / "parquet" / "cn" / "_latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    manifest_path = Path(latest["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    latest["coverage"] = coverage
    manifest["coverage"] = coverage
    latest_path.write_text(json.dumps(latest), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    store = MarketDataStore(market="CN", data_root=tmp_path)
    assert store.validate_latest()["status"] == "passed"

    forged = json.loads(evidence_path.read_text(encoding="utf-8"))
    forged["query_params"] = {"trade_date": "20260314"}
    unsigned = dict(forged)
    unsigned.pop("payload_sha256")
    forged["payload_sha256"] = canonical_json_sha256(unsigned)
    evidence_path.write_text(json.dumps(forged), encoding="utf-8")
    coverage["verified_nontrading_evidence_sha256"] = file_sha256(evidence_path)
    latest["coverage"] = coverage
    manifest["coverage"] = coverage
    latest_path.write_text(json.dumps(latest), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    validation = MarketDataStore(market="CN", data_root=tmp_path).validate_latest()
    assert validation["status"] == "failed"
    assert "coverage_nontrading_evidence_semantic:query_params_mismatch" in validation["blockers"]


def test_parquet_maintainer_excludes_symbol_on_exact_delist_date():
    class _Provider:
        def stock_basic(self, **_kwargs):
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "list_date": "20200101",
                        "delist_date": "20260707",
                        "list_status": "D",
                    }
                ]
            )

    maintainer = download_module.CNParquetBatchMaintainer.__new__(
        download_module.CNParquetBatchMaintainer
    )
    maintainer.downloader = type("_Downloader", (), {"pro": _Provider()})()

    assert maintainer._load_inactive_symbols("20260707", {"000001.SZ"}) == {"000001.SZ"}


def test_parquet_maintainer_uses_bak_daily_zero_as_evidence_only(tmp_path):
    class _Provider:
        def bak_daily(self, **kwargs):
            assert kwargs == {"trade_date": "20260707"}
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "20260707",
                        "open": 0.0,
                        "high": 0.0,
                        "low": 0.0,
                        "close": 10.0,
                        "pre_close": 10.0,
                        "change": 0.0,
                        "pct_chg": 0.0,
                        "vol": 0.0,
                        "amount": 0.0,
                    }
                ]
            )

    pit_path = tmp_path / "stock_basic_membership.parquet"
    pit_path.write_bytes(b"pit")
    maintainer = download_module.CNParquetBatchMaintainer.__new__(
        download_module.CNParquetBatchMaintainer
    )
    maintainer.downloader = type("_Downloader", (), {"pro": _Provider()})()
    maintainer.data_dir = tmp_path / "audit"
    payload = maintainer._load_verified_nontrading_bak_daily_zero(
        "20260707",
        {"000001.SZ"},
        pit_binding={
            "path": str(pit_path),
            "sha256": file_sha256(pit_path),
            "records": {
                "000001.SZ": PITUniverseRecord(
                    symbol="000001.SZ",
                    list_date="20200101",
                    source_list_status="L",
                )
            },
            "blockers": [],
        },
    )

    assert payload["verified_symbols"] == ["000001.SZ"]
    assert payload["writes_synthetic_bars"] is False
    assert payload["regulatory_suspension_claimed"] is False
    assert not list(tmp_path.rglob("bars.parquet"))


def test_parquet_maintainer_requeries_cached_empty_bak_daily_page(tmp_path):
    class _Provider:
        calls = 0

        def bak_daily(self, **kwargs):
            self.calls += 1
            assert kwargs == {"trade_date": "20260707"}
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "20260707",
                        "open": 0.0,
                        "high": 0.0,
                        "low": 0.0,
                        "close": 10.0,
                        "pre_close": 10.0,
                        "change": 0.0,
                        "pct_chg": 0.0,
                        "vol": 0.0,
                        "amount": 0.0,
                    }
                ]
            )

    pit_path = tmp_path / "stock_basic_membership.parquet"
    pit_path.write_bytes(b"pit")
    pit_sha256 = file_sha256(pit_path)
    data_dir = tmp_path / "audit"
    cache_path = evidence_cache_path(
        data_dir,
        trade_date="20260707",
        primary_missing_symbols=["000001.SZ"],
        pit_membership_sha256=pit_sha256,
    )
    empty_payload = build_bak_daily_nontrading_evidence(
        pd.DataFrame(),
        trade_date="20260707",
        primary_missing_symbols=["000001.SZ"],
        query_params={"trade_date": "20260707"},
        pit_membership_path=str(pit_path),
        pit_membership_sha256=pit_sha256,
    )
    write_evidence_cache(cache_path, empty_payload)

    provider = _Provider()
    maintainer = download_module.CNParquetBatchMaintainer.__new__(
        download_module.CNParquetBatchMaintainer
    )
    maintainer.downloader = type("_Downloader", (), {"pro": provider})()
    maintainer.data_dir = data_dir
    payload = maintainer._load_verified_nontrading_bak_daily_zero(
        "20260707",
        {"000001.SZ"},
        pit_binding={
            "path": str(pit_path),
            "sha256": pit_sha256,
            "records": {
                "000001.SZ": PITUniverseRecord(
                    symbol="000001.SZ",
                    list_date="20200101",
                    source_list_status="L",
                )
            },
            "blockers": [],
        },
    )

    assert provider.calls == 1
    assert payload["status"] == "queried"
    assert payload["verified_symbols"] == ["000001.SZ"]


def test_historical_upsert_requires_verified_latest_coverage(tmp_path):
    _write_seed_snapshot(tmp_path)
    store = MarketDataStore(market="CN", data_root=tmp_path)
    latest_path = tmp_path / "parquet" / "cn" / "_latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    latest["coverage"] = {**latest["coverage"], "complete": False}
    latest_path.write_text(json.dumps(latest), encoding="utf-8")
    manifest_path = Path(latest["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["coverage"] = latest["coverage"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    before = latest_path.read_text(encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="historical_upsert_requires_verified_latest_coverage",
    ):
        store.upsert_bars(
            pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "20260314",
                        "open": 9.0,
                        "high": 9.6,
                        "low": 8.9,
                        "close": 9.4,
                        "vol": 800,
                        "amount": 8200.0,
                        "adj_factor": 1.0,
                    }
                ]
            ),
            target_trade_date="20260314",
            source="unit-test",
            snapshot_id="historical-rejected",
        )

    assert latest_path.read_text(encoding="utf-8") == before


def test_same_date_republish_repairs_provenance_only_blocker(tmp_path):
    _write_seed_snapshot(tmp_path)
    latest_path = tmp_path / "parquet" / "cn" / "_latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    old_coverage = dict(latest["coverage"])
    latest["coverage"] = old_coverage
    latest_path.write_text(json.dumps(latest), encoding="utf-8")
    old_manifest_path = Path(latest["manifest_path"])
    old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
    old_manifest["coverage"] = old_coverage
    old_manifest["historical_scope_hash_backfilled"] = True
    old_manifest_path.write_text(json.dumps(old_manifest), encoding="utf-8")

    store = MarketDataStore(market="CN", data_root=tmp_path)
    result = store.upsert_bars(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20260315",
                    "open": 9.5,
                    "high": 10.0,
                    "low": 9.0,
                    "close": 9.8,
                    "vol": 900,
                    "amount": 9000.0,
                    "adj_factor": 1.0,
                },
                {
                    "ts_code": "000002.SZ",
                    "trade_date": "20260315",
                    "open": 19.5,
                    "high": 20.0,
                    "low": 19.0,
                    "close": 19.8,
                    "vol": 1900,
                    "amount": 19000.0,
                    "adj_factor": 1.0,
                },
            ]
        ),
        target_trade_date="20260315",
        source="unit-test-exact-date-republish",
        snapshot_id="same-date-repaired",
        metadata={
            "latest_available_trade_date": "20260315",
            "latest_complete_trade_date": "20260315",
            "coverage": {
                **old_coverage,
            },
        },
    )

    assert result["snapshot_id"] == "same-date-repaired"
    assert result.get("historical_scope_hash_backfilled") is not True
    assert result["coverage"]["expected_scope_sha256"] == old_coverage["expected_scope_sha256"]
    assert store.validate_latest()["status"] == "passed"


def test_upsert_rolls_back_pointer_and_roots_when_post_validation_fails(
    monkeypatch,
    tmp_path,
):
    _write_seed_snapshot(tmp_path)
    latest_path = tmp_path / "parquet" / "cn" / "_latest.json"
    pointer_payload = json.loads(latest_path.read_text(encoding="utf-8"))
    before_pointer = (
        json.dumps(
            pointer_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(", ", " : "),
        )
        + "\n\n"
    ).encode("utf-8")
    latest_path.write_bytes(before_pointer)
    before_pointer_sha256 = hashlib.sha256(before_pointer).hexdigest()
    store = MarketDataStore(market="CN", data_root=tmp_path)
    table_path = tmp_path / "parquet" / "cn" / "bars" / "year=2026" / "month=03" / "part.parquet"
    before_table = pd.read_parquet(table_path)
    original_gate = store.reader.clean_snapshot_gate
    gate_calls = 0

    def _fail_postcommit_gate(*, refresh=False):
        nonlocal gate_calls
        gate_calls += 1
        if gate_calls == 1:
            return original_gate(refresh=refresh)
        return {
            "healthy": False,
            "blockers": ["forced-post-check"],
            "snapshot_id": "post-check-rejected",
            "latest_complete_trade_date": "20260316",
        }

    monkeypatch.setattr(
        store.reader,
        "clean_snapshot_gate",
        _fail_postcommit_gate,
    )

    with pytest.raises(ValueError, match="post_commit_bars_validation_failed"):
        store.upsert_bars(
            pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "20260316",
                        "open": 10.0,
                        "high": 10.6,
                        "low": 9.9,
                        "close": 10.4,
                        "vol": 1000,
                        "amount": 12000.0,
                        "adj_factor": 1.1,
                    }
                ]
            ),
            target_trade_date="20260316",
            source="unit-test",
            snapshot_id="post-check-rejected",
            expected_latest_pointer_sha256=before_pointer_sha256,
        )

    assert latest_path.read_bytes() == before_pointer
    assert hashlib.sha256(latest_path.read_bytes()).hexdigest() == (before_pointer_sha256)
    pd.testing.assert_frame_equal(pd.read_parquet(table_path), before_table)


def test_upsert_bars_postcommit_does_not_depend_on_stale_macro_generation(
    monkeypatch,
    tmp_path,
):
    _write_seed_snapshot(tmp_path)
    latest_path = tmp_path / "parquet" / "cn" / "_latest.json"
    pointer_sha256 = hashlib.sha256(latest_path.read_bytes()).hexdigest()
    store = MarketDataStore(market="CN", data_root=tmp_path)

    def _full_validation_must_not_run():
        raise AssertionError("bars postcommit must not depend on cross-table Macro readiness")

    monkeypatch.setattr(store, "validate_latest", _full_validation_must_not_run)
    result = store.upsert_bars(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20260316",
                    "open": 10.0,
                    "high": 10.6,
                    "low": 9.9,
                    "close": 10.4,
                    "vol": 1000,
                    "amount": 12000.0,
                    "adj_factor": 1.1,
                }
            ]
        ),
        target_trade_date="20260316",
        source="unit-test-macro-cutover",
        snapshot_id="bars-before-retired-macro",
        expected_latest_pointer_sha256=pointer_sha256,
        metadata={
            "status": "OK",
            "latest_available_trade_date": "20260316",
            "latest_complete_trade_date": "20260316",
            "coverage": strict_coverage_v4(
                tmp_path,
                ["000001.SZ", "000002.SZ"],
                trade_date="20260316",
            ),
        },
    )

    assert result["snapshot_id"] == "bars-before-retired-macro"
    pointer = json.loads(latest_path.read_text(encoding="utf-8"))
    assert pointer["latest_complete_trade_date"] == "20260316"


def test_upsert_does_not_overwrite_external_pointer_on_precommit_cas_race(
    monkeypatch,
    tmp_path,
):
    _write_seed_snapshot(tmp_path)
    latest_path = tmp_path / "parquet" / "cn" / "_latest.json"
    expected_sha256 = hashlib.sha256(latest_path.read_bytes()).hexdigest()
    external_payload = json.loads(latest_path.read_text(encoding="utf-8"))
    external_payload["external_writer_marker"] = "preserve-me"
    external_pointer = (
        json.dumps(external_payload, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    store = MarketDataStore(market="CN", data_root=tmp_path)
    real_copy = store._copytree_hardlink_or_copy
    copy_count = 0

    def racing_copy(source, target):
        nonlocal copy_count
        real_copy(source, target)
        copy_count += 1
        if copy_count == 1:
            latest_path.write_bytes(external_pointer)

    monkeypatch.setattr(store, "_copytree_hardlink_or_copy", racing_copy)

    with pytest.raises(ValueError, match="market_pointer_cas_mismatch"):
        store.upsert_bars(
            pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "20260316",
                        "open": 10.0,
                        "high": 10.6,
                        "low": 9.9,
                        "close": 10.4,
                        "vol": 1000,
                        "amount": 12000.0,
                        "adj_factor": 1.1,
                    }
                ]
            ),
            target_trade_date="20260316",
            source="unit-test-cas-race",
            snapshot_id="cas-race-rejected",
            expected_latest_pointer_sha256=expected_sha256,
        )

    assert latest_path.read_bytes() == external_pointer


def test_snapshot_gate_fails_closed_when_manifest_json_is_unreadable(tmp_path):
    _write_seed_snapshot(tmp_path)
    manifest_path = tmp_path / "parquet" / "cn" / "_snapshots" / "seed.json"
    manifest_path.write_text("{not-json", encoding="utf-8")

    store = MarketDataStore(market="CN", data_root=tmp_path)
    gate = store.reader.clean_snapshot_gate(refresh=True)
    validation = store.validate_latest()

    assert gate["healthy"] is False
    assert any(blocker.startswith("manifest unreadable:") for blocker in gate["blockers"])
    assert validation["status"] == "failed"


def test_storage_validate_rejects_pointer_manifest_coverage_mismatch(tmp_path):
    _write_seed_snapshot(tmp_path)
    latest_path = tmp_path / "parquet" / "cn" / "_latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    latest["coverage"]["symbol_count"] = 3
    latest_path.write_text(json.dumps(latest), encoding="utf-8")

    store = MarketDataStore(market="CN", data_root=tmp_path)
    validation = store.validate_latest()

    assert validation["status"] == "failed"
    assert any(
        blocker.startswith("coverage_pointer_manifest_mismatch:")
        for blocker in validation["blockers"]
    )


@pytest.mark.parametrize(
    ("patch", "expected_blocker"),
    [
        ({"blocking_incomplete_count": 1}, "coverage_blocking_incomplete_count_nonzero:1"),
        ({"expected_scope_count": 0}, "coverage_expected_scope_count_missing_or_nonpositive"),
        ({"coverage_complete_count": 1}, "coverage_complete_count_mismatch:1!=2"),
        ({"coverage_ratio": 0.5}, "coverage_ratio_not_one"),
    ],
)
def test_storage_validate_rejects_incoherent_complete_coverage(
    tmp_path,
    patch,
    expected_blocker,
):
    _write_seed_snapshot(tmp_path)
    latest_path = tmp_path / "parquet" / "cn" / "_latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    coverage = {
        "complete": True,
        "coverage_trade_date": "20260315",
        "expected_scope_sha256": "a" * 64,
        "blocking_incomplete_count": 0,
        "expected_scope_count": 2,
        "coverage_complete_count": 2,
        "coverage_ratio": 1.0,
    }
    coverage.update(patch)
    latest["coverage"] = coverage
    latest_path.write_text(json.dumps(latest), encoding="utf-8")
    manifest_path = Path(latest["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["coverage"] = coverage
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    validation = MarketDataStore(market="CN", data_root=tmp_path).validate_latest()

    assert validation["status"] == "failed"
    assert expected_blocker in validation["blockers"]


def test_storage_validate_rejects_unverified_allowed_stale_symbols(tmp_path):
    _write_seed_snapshot(tmp_path)
    latest_path = tmp_path / "parquet" / "cn" / "_latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    coverage = {
        "complete": True,
        "coverage_trade_date": "20260315",
        "expected_scope_sha256": "a" * 64,
        "blocking_incomplete_count": 0,
        "expected_scope_count": 2,
        "coverage_complete_count": 2,
        "coverage_ratio": 1.0,
        "allowed_stale_symbols": ["000002.SZ"],
    }
    latest["coverage"] = coverage
    latest_path.write_text(json.dumps(latest), encoding="utf-8")
    manifest_path = Path(latest["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["coverage"] = coverage
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    validation = MarketDataStore(market="CN", data_root=tmp_path).validate_latest()

    assert validation["status"] == "failed"
    assert "coverage_unverified_allowed_stale_symbols_not_permitted" in (validation["blockers"])


def test_storage_validate_rejects_overlapping_v2_classification_sets(tmp_path):
    _write_seed_snapshot(tmp_path)
    latest_path = tmp_path / "parquet" / "cn" / "_latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    coverage = {
        "coverage_schema_version": "cn-full-a-coverage.v2",
        "complete": True,
        "coverage_trade_date": "20260315",
        "expected_scope_sha256": "a" * 64,
        "blocking_incomplete_count": 0,
        "expected_scope_count": 2,
        "coverage_complete_count": 2,
        "coverage_ratio": 1.0,
        "observed_bar_count": 1,
        "suspended_symbols": ["000002.SZ"],
        "inactive_symbols": ["000002.SZ"],
        "allowed_stale_symbols": [],
        "non_blocking_absent_symbols": ["000002.SZ"],
        "true_missing_symbols": [],
        "classification_sets_disjoint": True,
    }
    latest["coverage"] = coverage
    latest_path.write_text(json.dumps(latest), encoding="utf-8")
    manifest_path = Path(latest["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["coverage"] = coverage
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    validation = MarketDataStore(market="CN", data_root=tmp_path).validate_latest()

    assert validation["status"] == "failed"
    assert "coverage_classification_sets_not_disjoint" in validation["blockers"]


def test_upsert_bars_rolls_back_live_roots_when_latest_pointer_write_fails(monkeypatch, tmp_path):
    _write_seed_snapshot(tmp_path)
    store = MarketDataStore(market="CN", data_root=tmp_path)
    real_atomic_write_json = store._atomic_write_json

    def flaky_atomic_write_json(payload, path):
        if Path(path).name == "_latest.json":
            raise RuntimeError("pointer write failed")
        real_atomic_write_json(payload, path)

    monkeypatch.setattr(store, "_atomic_write_json", flaky_atomic_write_json)

    with pytest.raises(RuntimeError, match="pointer write failed"):
        store.upsert_bars(
            pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "20260316",
                        "open": 10.0,
                        "high": 10.6,
                        "low": 9.9,
                        "close": 10.4,
                        "vol": 1000,
                        "amount": 12000.0,
                        "adj_factor": 1.1,
                    }
                ]
            ),
            target_trade_date="20260316",
            source="unit-test",
            snapshot_id="rollback",
            metadata={
                "status": "OK",
                "latest_available_trade_date": "20260316",
                "latest_complete_trade_date": "20260316",
            },
        )

    latest = json.loads((tmp_path / "parquet" / "cn" / "_latest.json").read_text(encoding="utf-8"))
    table = pd.read_parquet(
        tmp_path / "parquet" / "cn" / "bars" / "year=2026" / "month=03" / "part.parquet"
    )
    serving = pd.read_parquet(
        tmp_path / "parquet_serving" / "cn" / "bars" / "symbol=000001.SZ" / "bars.parquet"
    )
    assert latest["snapshot_id"] == "seed"
    assert set(table["trade_date"]) == {"20260315"}
    assert set(serving["trade_date"]) == {"20260315"}
    assert not (tmp_path / "parquet_staging" / "cn" / "rollback").exists()
    assert store.validate_latest()["status"] == "passed"


def test_run_market_maintenance_parquet_direct_upserts_and_writes_audit_artifacts(
    monkeypatch, tmp_path
):
    _write_seed_snapshot(tmp_path)
    audit_root = tmp_path / "cn_market_full"

    class FakePro:
        def stock_basic(self, **_kwargs):
            return pd.DataFrame(
                [
                    {"ts_code": "000001.SZ", "list_date": "20200101"},
                    {"ts_code": "000002.SZ", "list_date": "20200101"},
                ]
            )

        def trade_cal(self, **_kwargs):
            return pd.DataFrame(
                [
                    {"cal_date": "20260315", "is_open": 1},
                    {"cal_date": "20260316", "is_open": 1},
                ]
            )

        def suspend_d(self, **_kwargs):
            return pd.DataFrame()

        def daily(self, trade_date=None, **_kwargs):
            assert trade_date == "20260316"
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "20260316",
                        "open": 10.0,
                        "high": 10.6,
                        "low": 9.9,
                        "close": 10.4,
                        "pre_close": 9.8,
                        "change": 0.6,
                        "pct_chg": 6.12,
                        "vol": 1000,
                        "amount": 12000.0,
                    },
                    {
                        "ts_code": "000002.SZ",
                        "trade_date": "20260316",
                        "open": 20.0,
                        "high": 20.8,
                        "low": 19.8,
                        "close": 20.4,
                        "pre_close": 20.0,
                        "change": 0.4,
                        "pct_chg": 2.0,
                        "vol": 2000,
                        "amount": 22000.0,
                    },
                ]
            )

        def adj_factor(self, trade_date=None, **_kwargs):
            assert trade_date == "20260316"
            return pd.DataFrame(
                [
                    {"ts_code": "000001.SZ", "trade_date": "20260316", "adj_factor": 1.1},
                    {"ts_code": "000002.SZ", "trade_date": "20260316", "adj_factor": 1.0},
                ]
            )

        def daily_basic(self, trade_date=None, **_kwargs):
            assert trade_date == "20260316"
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "20260316",
                        "turnover_rate": 1.2,
                        "volume_ratio": 1.1,
                        "pe": 12.0,
                        "pb": 1.5,
                        "total_mv": 100000.0,
                        "circ_mv": 90000.0,
                    }
                ]
            )

    monkeypatch.setattr(download_module.config, "TUSHARE_TOKEN", "dummy-token")
    monkeypatch.setattr(download_module.config, "TUSHARE_URL", "http://example.invalid")
    monkeypatch.setattr(
        download_module.config, "MARKET_DATA_BASE_DIR", str(tmp_path), raising=False
    )
    monkeypatch.setattr(download_module.config, "CN_FRESHNESS_MODE", "strict")
    monkeypatch.setattr(
        download_cn_module, "create_tushare_pro", lambda *_args, **_kwargs: FakePro()
    )
    generation = _write_pit_generation(tmp_path)

    result = download_module.run_market_maintenance(
        market="CN",
        categories=["full_a"],
        storage_mode="parquet-direct",
        data_dir=str(audit_root),
        **_production_binding_args(tmp_path, generation),
    )

    assert result["storage_mode"] == "parquet-direct"
    assert result["parquet_commit"]["status"] == "OK"
    assert result["parquet_commit"]["latest_complete_trade_date"] == "20260316"
    assert not list(audit_root.glob("*/*.csv"))
    progress = json.loads((audit_root / "progress_summary.json").read_text(encoding="utf-8"))
    failed = json.loads((audit_root / "failed_batches.json").read_text(encoding="utf-8"))
    assert progress["storage_mode"] == "parquet-direct"
    assert progress["status"] == "OK"
    assert progress["daily_basic_coverage"]["coverage_ratio"] == 0.5
    assert failed["failed_batch_count"] == 0

    blocked = download_module.run_market_maintenance(
        market="CN",
        categories=["full_a"],
        storage_mode="parquet-direct",
        data_dir=str(audit_root),
        allowed_stale_symbols=["000001.SZ"],
        **_production_binding_args(tmp_path, generation),
    )

    assert blocked["parquet_commit"]["status"] == "BLOCKED"
    assert "unverified_allowed_stale_symbols_not_permitted" in (blocked["completeness"]["blockers"])
    assert blocked["completeness"]["true_missing_symbols"] == []
    assert blocked["completeness"]["allowed_stale_symbols"] == []
    assert blocked["completeness"]["requested_allowed_stale_symbols"] == ["000001.SZ"]
    assert blocked["completeness"]["requested_allowed_absent_symbols"] == []


def test_parquet_direct_explicit_historical_target_preserves_latest_pointer(monkeypatch, tmp_path):
    _write_seed_snapshot(tmp_path)
    latest_path = tmp_path / "parquet" / "cn" / "_latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    latest["coverage"] = dict(latest["coverage"])
    latest_path.write_text(json.dumps(latest), encoding="utf-8")
    manifest_path = Path(latest["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["coverage"] = latest["coverage"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    audit_root = tmp_path / "cn_market_full"

    class FakePro:
        def stock_basic(self, **_kwargs):
            return pd.DataFrame(
                [
                    {"ts_code": "000001.SZ", "list_date": "20200101"},
                    {"ts_code": "000002.SZ", "list_date": "20200101"},
                ]
            )

        def trade_cal(self, **_kwargs):
            return pd.DataFrame(
                [
                    {"cal_date": "20260314", "is_open": 1},
                    {"cal_date": "20260315", "is_open": 1},
                ]
            )

        def suspend_d(self, **_kwargs):
            return pd.DataFrame()

        def daily(self, trade_date=None, **_kwargs):
            assert trade_date == "20260314"
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "20260314",
                        "open": 9.0,
                        "high": 9.6,
                        "low": 8.9,
                        "close": 9.4,
                        "pre_close": 9.2,
                        "change": 0.2,
                        "pct_chg": 2.17,
                        "vol": 800,
                        "amount": 8200.0,
                    },
                    {
                        "ts_code": "000002.SZ",
                        "trade_date": "20260314",
                        "open": 19.0,
                        "high": 19.8,
                        "low": 18.8,
                        "close": 19.4,
                        "pre_close": 19.2,
                        "change": 0.2,
                        "pct_chg": 1.04,
                        "vol": 1800,
                        "amount": 18200.0,
                    },
                ]
            )

        def adj_factor(self, trade_date=None, **_kwargs):
            assert trade_date == "20260314"
            return pd.DataFrame(
                [
                    {"ts_code": "000001.SZ", "trade_date": "20260314", "adj_factor": 1.1},
                    {"ts_code": "000002.SZ", "trade_date": "20260314", "adj_factor": 1.0},
                ]
            )

        def daily_basic(self, trade_date=None, **_kwargs):
            assert trade_date == "20260314"
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "20260314",
                        "turnover_rate": 1.0,
                        "volume_ratio": 1.0,
                        "pe": 12.0,
                        "pb": 1.5,
                        "total_mv": 100000.0,
                        "circ_mv": 90000.0,
                    },
                    {
                        "ts_code": "000002.SZ",
                        "trade_date": "20260314",
                        "turnover_rate": 1.0,
                        "volume_ratio": 1.0,
                        "pe": 10.0,
                        "pb": 1.2,
                        "total_mv": 200000.0,
                        "circ_mv": 190000.0,
                    },
                ]
            )

    monkeypatch.setattr(download_module.config, "TUSHARE_TOKEN", "dummy-token")
    monkeypatch.setattr(download_module.config, "TUSHARE_URL", "http://example.invalid")
    monkeypatch.setattr(
        download_module.config, "MARKET_DATA_BASE_DIR", str(tmp_path), raising=False
    )
    monkeypatch.setattr(download_module.config, "CN_FRESHNESS_MODE", "strict")
    monkeypatch.setattr(
        download_cn_module, "create_tushare_pro", lambda *_args, **_kwargs: FakePro()
    )
    generation = _write_pit_generation(tmp_path)

    result = download_module.run_market_maintenance(
        market="CN",
        categories=["full_a"],
        storage_mode="parquet-direct",
        target_date="20260314",
        data_dir=str(audit_root),
        **_production_binding_args(tmp_path, generation),
    )

    assert result["completeness"]["effective_target_trade_date"] == "20260314"
    assert result["parquet_commit"]["status"] == "OK"
    assert result["parquet_commit"]["latest_complete_trade_date"] == "20260315"
    latest = json.loads((tmp_path / "parquet" / "cn" / "_latest.json").read_text(encoding="utf-8"))
    assert latest["latest_complete_trade_date"] == "20260315"


def test_parquet_direct_historical_target_excludes_prelisting_symbols(monkeypatch, tmp_path):
    _write_seed_snapshot(tmp_path)
    latest_path = tmp_path / "parquet" / "cn" / "_latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    latest["coverage"] = strict_coverage_v4(
        tmp_path,
        ["000001.SZ", "000002.SZ", "000003.SZ"],
        trade_date="20260315",
    )
    latest_path.write_text(json.dumps(latest), encoding="utf-8")
    manifest_path = Path(latest["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["coverage"] = latest["coverage"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    audit_root = tmp_path / "cn_market_full"

    class FakePro:
        def stock_basic(self, list_status="L", **_kwargs):
            if list_status != "L":
                return pd.DataFrame(
                    columns=["ts_code", "name", "list_status", "list_date", "delist_date"]
                )
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "name": "One",
                        "list_status": "L",
                        "list_date": "20200101",
                        "delist_date": "",
                    },
                    {
                        "ts_code": "000002.SZ",
                        "name": "Two",
                        "list_status": "L",
                        "list_date": "20200101",
                        "delist_date": "",
                    },
                    {
                        "ts_code": "000003.SZ",
                        "name": "Future",
                        "list_status": "L",
                        "list_date": "20260316",
                        "delist_date": "",
                    },
                ]
            )

        def trade_cal(self, **_kwargs):
            return pd.DataFrame(
                [
                    {"cal_date": "20260314", "is_open": 1},
                    {"cal_date": "20260315", "is_open": 1},
                ]
            )

        def suspend_d(self, **_kwargs):
            return pd.DataFrame()

        def daily(self, trade_date=None, **_kwargs):
            assert trade_date == "20260314"
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "20260314",
                        "open": 9.0,
                        "high": 9.6,
                        "low": 8.9,
                        "close": 9.4,
                        "pre_close": 9.2,
                        "change": 0.2,
                        "pct_chg": 2.17,
                        "vol": 800,
                        "amount": 8200.0,
                    },
                    {
                        "ts_code": "000002.SZ",
                        "trade_date": "20260314",
                        "open": 19.0,
                        "high": 19.8,
                        "low": 18.8,
                        "close": 19.4,
                        "pre_close": 19.2,
                        "change": 0.2,
                        "pct_chg": 1.04,
                        "vol": 1800,
                        "amount": 18200.0,
                    },
                ]
            )

        def adj_factor(self, trade_date=None, **_kwargs):
            assert trade_date == "20260314"
            return pd.DataFrame(
                [
                    {"ts_code": "000001.SZ", "trade_date": "20260314", "adj_factor": 1.1},
                    {"ts_code": "000002.SZ", "trade_date": "20260314", "adj_factor": 1.0},
                ]
            )

        def daily_basic(self, trade_date=None, **_kwargs):
            assert trade_date == "20260314"
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "20260314",
                        "turnover_rate": 1.0,
                        "volume_ratio": 1.0,
                        "pe": 12.0,
                        "pb": 1.5,
                        "total_mv": 100000.0,
                        "circ_mv": 90000.0,
                    },
                    {
                        "ts_code": "000002.SZ",
                        "trade_date": "20260314",
                        "turnover_rate": 1.0,
                        "volume_ratio": 1.0,
                        "pe": 10.0,
                        "pb": 1.2,
                        "total_mv": 200000.0,
                        "circ_mv": 190000.0,
                    },
                ]
            )

    monkeypatch.setattr(download_module.config, "TUSHARE_TOKEN", "dummy-token")
    monkeypatch.setattr(download_module.config, "TUSHARE_URL", "http://example.invalid")
    monkeypatch.setattr(
        download_module.config, "MARKET_DATA_BASE_DIR", str(tmp_path), raising=False
    )
    monkeypatch.setattr(download_module.config, "CN_FRESHNESS_MODE", "strict")
    monkeypatch.setattr(
        download_cn_module, "create_tushare_pro", lambda *_args, **_kwargs: FakePro()
    )
    generation = _write_pit_generation(
        tmp_path,
        records=[
            PITUniverseRecord(
                symbol="000001.SZ",
                name="One",
                list_date="20200101",
                source_list_status="L",
                observed_at="2026-03-16T00:00:00Z",
                source_run_id="unit-pit",
            ),
            PITUniverseRecord(
                symbol="000002.SZ",
                name="Two",
                list_date="20200101",
                source_list_status="L",
                observed_at="2026-03-16T00:00:00Z",
                source_run_id="unit-pit",
            ),
            PITUniverseRecord(
                symbol="000003.SZ",
                name="Future",
                list_date="20260316",
                source_list_status="L",
                observed_at="2026-03-16T00:00:00Z",
                source_run_id="unit-pit",
            ),
        ],
    )

    result = download_module.run_market_maintenance(
        market="CN",
        categories=["full_a"],
        storage_mode="parquet-direct",
        target_date="20260314",
        data_dir=str(audit_root),
        **_production_binding_args(tmp_path, generation),
    )

    assert result["parquet_commit"]["status"] == "OK"
    assert result["parquet_commit"]["historical_upsert_coverage_preserved"] is True
    target_coverage = result["parquet_commit"]["historical_upsert_target_coverage"]
    assert target_coverage["expected_scope_count"] == 3
    assert target_coverage["coverage_complete_count"] == 3
    assert target_coverage["inactive_symbols"] == ["000003.SZ"]
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    assert latest["coverage"]["expected_scope_sha256"] == symbol_set_sha256(
        ["000001.SZ", "000002.SZ", "000003.SZ"]
    )
    validation = MarketDataStore(market="CN", data_root=tmp_path).validate_latest()
    assert validation["status"] == "passed"
    table = pd.read_parquet(Path(latest["table_root"]) / "year=2026" / "month=03" / "part.parquet")
    assert set(table.loc[table["trade_date"].eq("20260314"), "ts_code"]) == {
        "000001.SZ",
        "000002.SZ",
    }
