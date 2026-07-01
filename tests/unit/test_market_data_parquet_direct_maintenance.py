from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import quant_investor.market.download as download_module
import quant_investor.market.download_cn as download_cn_module
from quant_investor.market.market_data_store import MarketDataStore


def test_cn_legacy_maintenance_is_disabled_outside_staged():
    with pytest.raises(ValueError, match="CN legacy CSV maintenance is disabled"):
        download_module.run_market_maintenance(
            market="CN",
            categories=["full_a"],
            storage_mode="legacy",
        )


def test_cn_auto_maintenance_uses_parquet_direct(monkeypatch):
    captured: dict[str, object] = {}

    class FakeMaintainer:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def maintain(self, **kwargs):
            captured["maintain"] = kwargs
            return {"storage_mode": "parquet-direct"}

    monkeypatch.setattr(download_module, "CNParquetBatchMaintainer", FakeMaintainer)

    result = download_module.run_market_maintenance(
        market="CN",
        categories=["full_a"],
        target_date="20260316",
    )

    assert result["storage_mode"] == "parquet-direct"
    assert captured["maintain"]["categories"] == ["full_a"]
    assert captured["maintain"]["target_date"] == "20260316"


def _write_seed_snapshot(root: Path) -> None:
    table_root = root / "parquet" / "cn" / "bars"
    serving_root = root / "parquet_serving" / "cn" / "bars"
    manifest_path = root / "parquet" / "cn" / "_snapshots" / "seed.json"
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
    frame[frame["ts_code"].eq("000001.SZ")].to_parquet(symbol_1 / "bars.parquet", index=False)
    frame[frame["ts_code"].eq("000002.SZ")].to_parquet(symbol_2 / "bars.parquet", index=False)
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "snapshot_id": "seed",
                "status": "OK",
                "table_root": str(table_root),
                "derived_serving_root": str(serving_root),
                "latest_complete_trade_date": "20260315",
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
                "coverage": {"row_count": 2, "symbol_count": 2},
                "blockers": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_upsert_bars_merges_target_day_without_replacing_history(tmp_path):
    _write_seed_snapshot(tmp_path)
    store = MarketDataStore(market="CN", data_root=tmp_path)

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
        metadata={
            "status": "OK",
            "latest_available_trade_date": "20260316",
            "latest_complete_trade_date": "20260316",
        },
    )

    table = pd.read_parquet(tmp_path / "parquet" / "cn" / "bars" / "year=2026" / "month=03" / "part.parquet")
    assert manifest["snapshot_id"] == "upserted"
    assert manifest["row_count"] == 4
    assert set(table["ts_code"]) == {"000001.SZ", "000002.SZ", "000003.SZ"}
    assert set(table["trade_date"]) == {"20260315", "20260316"}
    assert json.loads((tmp_path / "parquet" / "cn" / "_latest.json").read_text())["snapshot_id"] == "upserted"
    assert store.validate_latest()["status"] == "passed"


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
    table = pd.read_parquet(tmp_path / "parquet" / "cn" / "bars" / "year=2026" / "month=03" / "part.parquet")
    serving = pd.read_parquet(tmp_path / "parquet_serving" / "cn" / "bars" / "symbol=000001.SZ" / "bars.parquet")
    assert latest["snapshot_id"] == "seed"
    assert set(table["trade_date"]) == {"20260315"}
    assert set(serving["trade_date"]) == {"20260315"}
    assert not (tmp_path / "parquet_staging" / "cn" / "rollback").exists()
    assert store.validate_latest()["status"] == "passed"


def test_run_market_maintenance_parquet_direct_upserts_and_writes_audit_artifacts(monkeypatch, tmp_path):
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
    monkeypatch.setattr(download_module.config, "MARKET_DATA_BASE_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(download_module.config, "CN_FRESHNESS_MODE", "strict")
    monkeypatch.setattr(download_cn_module, "create_tushare_pro", lambda *_args, **_kwargs: FakePro())

    result = download_module.run_market_maintenance(
        market="CN",
        categories=["full_a"],
        storage_mode="parquet-direct",
        data_dir=str(audit_root),
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


def test_parquet_direct_explicit_historical_target_preserves_latest_pointer(monkeypatch, tmp_path):
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
    monkeypatch.setattr(download_module.config, "MARKET_DATA_BASE_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(download_module.config, "CN_FRESHNESS_MODE", "strict")
    monkeypatch.setattr(download_cn_module, "create_tushare_pro", lambda *_args, **_kwargs: FakePro())

    result = download_module.run_market_maintenance(
        market="CN",
        categories=["full_a"],
        storage_mode="parquet-direct",
        target_date="20260314",
        data_dir=str(audit_root),
    )

    assert result["completeness"]["effective_target_trade_date"] == "20260314"
    assert result["parquet_commit"]["status"] == "OK"
    assert result["parquet_commit"]["latest_complete_trade_date"] == "20260315"
    latest = json.loads((tmp_path / "parquet" / "cn" / "_latest.json").read_text(encoding="utf-8"))
    assert latest["latest_complete_trade_date"] == "20260315"
