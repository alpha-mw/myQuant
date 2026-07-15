from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from quant_investor.market.market_data_reader import (
    MarketDataReader,
    MarketDataUnavailableError,
)
from quant_investor.market.backtest import _load_market_frame
from quant_investor.market.market_data_store import MarketDataStore
from quant_investor.market.pit_universe import (
    LIST_STATUS_DELISTED,
    LIST_STATUS_LISTED,
    PITUniverseRecord,
    PITUniverseStore,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_parquet_fixture(root: Path, *, status: str = "OK") -> dict[str, Path]:
    canonical = root / "parquet" / "cn" / "bars" / "year=2026" / "month=01"
    serving_1 = root / "parquet_serving" / "cn" / "bars" / "symbol=000001.SZ"
    serving_2 = root / "parquet_serving" / "cn" / "bars" / "symbol=000002.SZ"
    snapshot_dir = root / "parquet" / "cn" / "_snapshots"
    universe_dir = root / "cn_universe"
    daily_basic_dir = root / "parquet" / "cn" / "daily_basic"
    canonical.mkdir(parents=True, exist_ok=True)
    serving_1.mkdir(parents=True, exist_ok=True)
    serving_2.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    universe_dir.mkdir(parents=True, exist_ok=True)
    daily_basic_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "trade_date": "20260101", "open": 9.5, "high": 10.2, "low": 9.3, "close": 10.0, "vol": 1000},
            {"ts_code": "000001.SZ", "trade_date": "20260102", "open": 10.0, "high": 11.2, "low": 9.8, "close": 11.0, "vol": 1100},
            {"ts_code": "000001.SZ", "trade_date": "20260103", "open": 11.0, "high": 12.2, "low": 10.8, "close": 12.0, "vol": 1200},
            {"ts_code": "000002.SZ", "trade_date": "20260103", "open": 20.0, "high": 21.2, "low": 19.8, "close": 21.0, "vol": 2100},
        ]
    )
    frame.to_parquet(canonical / "part.parquet", index=False)
    frame[frame["ts_code"].eq("000001.SZ")].to_parquet(serving_1 / "bars.parquet", index=False)
    frame[frame["ts_code"].eq("000002.SZ")].to_parquet(serving_2 / "bars.parquet", index=False)

    daily_basic = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "trade_date": "20260103", "total_mv": 100.0},
            {"ts_code": "000002.SZ", "trade_date": "20260103", "total_mv": 200.0},
            {"ts_code": "000001.SZ", "trade_date": "20260102", "total_mv": 90.0},
        ]
    )
    daily_basic.to_parquet(daily_basic_dir / "part.parquet", index=False)

    manifest = snapshot_dir / "snap-001.json"
    manifest.write_text(json.dumps({"snapshot_id": "snap-001"}, ensure_ascii=False), encoding="utf-8")
    latest = {
        "status": status,
        "snapshot_id": "snap-001",
        "latest_complete_trade_date": "20260103",
        "latest_trade_date": "20260103",
        "table_root": str(root / "parquet" / "cn" / "bars"),
        "derived_serving_root": str(root / "parquet_serving" / "cn" / "bars"),
        "manifest_path": str(manifest),
        "coverage": {"row_count": int(len(frame)), "symbol_count": 2},
        "blockers": [] if status == "OK" else ["fixture_blocker"],
    }
    latest_path = root / "parquet" / "cn" / "_latest.json"
    latest_path.write_text(json.dumps(latest, ensure_ascii=False), encoding="utf-8")

    catalog = {
        "schema_version": "fixture",
        "market": "CN",
        "latest_snapshot_id": "snap-001",
        "tables": {
            "daily_basic": {
                "logical_table": "daily_basic",
                "path": str(daily_basic_dir / "part.parquet"),
                "table_root": str(daily_basic_dir),
                "date_column": "trade_date",
                "columns": ["ts_code", "trade_date", "total_mv"],
                "status": "ok",
            }
        },
    }
    (root / "parquet" / "cn" / "_catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False),
        encoding="utf-8",
    )
    components = {
        "hs300": ["000001.SZ"],
        "zz500": ["000002.SZ"],
        "zz1000": [],
        "full_a": ["000001.SZ", "000002.SZ"],
        "stats": {"total_unique": 2},
    }
    (universe_dir / "cn_index_components.json").write_text(
        json.dumps(components, ensure_ascii=False),
        encoding="utf-8",
    )
    return {"latest": latest_path, "canonical": canonical, "serving": root / "parquet_serving" / "cn" / "bars"}


def test_strict_parquet_reader_reads_symbol_batch_cross_section_and_catalog_table(tmp_path: Path) -> None:
    _write_parquet_fixture(tmp_path)
    reader = MarketDataReader(market="CN", data_root=tmp_path)

    assert reader.latest_trade_date("full_a") == "20260103"
    assert reader.list_symbols("hs300") == ["000001.SZ"]

    single = reader.read_symbol_frame(
        "000001.SZ",
        start_date="20260102",
        end_date="20260103",
        columns=["trade_date", "close"],
    )
    assert single.metadata["backend"] == "parquet"
    assert single.metadata["fallback_used"] is False
    assert single.frame.to_dict(orient="records") == [
        {"trade_date": "20260102", "close": 11.0},
        {"trade_date": "20260103", "close": 12.0},
    ]

    batch = reader.read_symbol_frames(
        ["000001.SZ", "000002.SZ"],
        columns=["ts_code", "trade_date", "close"],
    )
    assert sorted(batch) == ["000001.SZ", "000002.SZ"]
    assert batch["000002.SZ"].frame.iloc[0]["close"] == 21.0

    cross_section = reader.read_cross_section(
        "20260103",
        columns=["ts_code", "trade_date", "close"],
    )
    assert set(cross_section["symbol"]) == {"000001.SZ", "000002.SZ"}
    assert set(cross_section["trade_date"]) == {"20260103"}

    daily_basic = reader.read_table(
        "daily_basic",
        as_of="20260103",
        columns=["ts_code", "trade_date", "total_mv"],
    )
    assert daily_basic["total_mv"].sum() == 300.0
    assert set(daily_basic["trade_date"]) == {"20260103"}


def test_strict_catalog_resolves_table_path_relative_to_parquet_market_root(
    tmp_path: Path,
) -> None:
    _write_parquet_fixture(tmp_path)
    generation_path = (
        tmp_path
        / "parquet"
        / "cn"
        / "macro_daily"
        / "_generations"
        / "macro-gen-001"
        / "macro_daily.parquet"
    )
    generation_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [{"trade_date": "20260103", "macro_score": 0.25}],
    ).to_parquet(generation_path, index=False)
    catalog_path = tmp_path / "parquet" / "cn" / "_catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "schema_version": "strict-parquet-catalog.v1",
                "market": "CN",
                "tables": {
                    "macro_daily": {
                        "path": (
                            "macro_daily/_generations/macro-gen-001/"
                            "macro_daily.parquet"
                        ),
                        "date_column": "trade_date",
                        "sha256": _sha256(generation_path),
                        "size_bytes": generation_path.stat().st_size,
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    frame = MarketDataReader(market="CN", data_root=tmp_path).read_table(
        "macro_daily",
        as_of="20260103",
    )

    assert frame.to_dict(orient="records") == [
        {"trade_date": "20260103", "macro_score": 0.25}
    ]


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({}, "hash missing"),
        ({"sha256": "0" * 64}, "hash mismatch"),
        (
            {"sha256": "0" * 64, "parquet_sha256": "1" * 64},
            "hash conflict",
        ),
        ({"size_bytes": 1}, "size mismatch"),
    ],
)
def test_strict_catalog_requires_consistent_hash_and_size(
    tmp_path: Path,
    metadata: dict[str, object],
    message: str,
) -> None:
    _write_parquet_fixture(tmp_path)
    table_path = tmp_path / "parquet" / "cn" / "daily_basic" / "part.parquet"
    entry: dict[str, object] = {
        "path": "daily_basic/part.parquet",
        "date_column": "trade_date",
        "sha256": _sha256(table_path),
        **metadata,
    }
    if not metadata:
        entry.pop("sha256")
    catalog_path = tmp_path / "parquet" / "cn" / "_catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "schema_version": "strict-parquet-catalog.v1",
                "tables": {"daily_basic": entry},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(MarketDataUnavailableError, match=message):
        MarketDataReader(market="CN", data_root=tmp_path).read_table(
            "daily_basic"
        )


def test_strict_catalog_blocks_path_replacement_during_same_fd_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_parquet_fixture(tmp_path)
    table_path = tmp_path / "parquet" / "cn" / "daily_basic" / "part.parquet"
    catalog_path = tmp_path / "parquet" / "cn" / "_catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "schema_version": "strict-parquet-catalog.v1",
                "tables": {
                    "daily_basic": {
                        "path": "daily_basic/part.parquet",
                        "date_column": "trade_date",
                        "sha256": _sha256(table_path),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    replacement = table_path.with_name("replacement.parquet")
    pd.DataFrame(
        [{"ts_code": "999999.SZ", "trade_date": "20260103"}]
    ).to_parquet(replacement, index=False)
    original_read = pd.read_parquet

    def _replace_after_fd_read(*args, **kwargs):
        frame = original_read(*args, **kwargs)
        os.replace(replacement, table_path)
        return frame

    monkeypatch.setattr(pd, "read_parquet", _replace_after_fd_read)
    with pytest.raises(MarketDataUnavailableError, match="replaced during read"):
        MarketDataReader(market="CN", data_root=tmp_path).read_table(
            "daily_basic"
        )


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({}, "path missing"),
        ({"path": "/tmp/macro_daily.parquet"}, "absolute path rejected"),
        (
            {"path": "macro_daily/../daily_basic/part.parquet"},
            "parent traversal rejected",
        ),
        (
            {"path": "macro_daily/_generations/missing.parquet"},
            "path missing or unreadable",
        ),
    ],
)
def test_strict_catalog_rejects_unsafe_or_missing_table_paths(
    tmp_path: Path,
    entry: dict[str, str],
    message: str,
) -> None:
    _write_parquet_fixture(tmp_path)
    catalog_path = tmp_path / "parquet" / "cn" / "_catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "schema_version": "strict-parquet-catalog.v1",
                "market": "CN",
                "tables": {
                    "macro_daily": {"date_column": "trade_date", **entry}
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    reader = MarketDataReader(market="CN", data_root=tmp_path)
    with pytest.raises(MarketDataUnavailableError, match=message):
        reader.read_table("macro_daily")


def test_strict_catalog_rejects_symlink_table_path(tmp_path: Path) -> None:
    _write_parquet_fixture(tmp_path)
    outside = tmp_path / "outside_macro_daily.parquet"
    pd.DataFrame(
        [{"trade_date": "20260103", "macro_score": 0.25}],
    ).to_parquet(outside, index=False)
    link_path = tmp_path / "parquet" / "cn" / "macro_daily.parquet"
    link_path.symlink_to(outside)
    catalog_path = tmp_path / "parquet" / "cn" / "_catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "schema_version": "strict-parquet-catalog.v1",
                "market": "CN",
                "tables": {
                    "macro_daily": {
                        "path": "macro_daily.parquet",
                        "date_column": "trade_date",
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    reader = MarketDataReader(market="CN", data_root=tmp_path)
    with pytest.raises(MarketDataUnavailableError, match="symlink rejected"):
        reader.read_table("macro_daily")


def test_reader_list_symbols_filters_by_pit_universe_only_when_enabled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_parquet_fixture(tmp_path)
    pit_store = PITUniverseStore(
        root_dir=tmp_path / "parquet" / "cn" / "reference",
        raw_root=tmp_path / "pit_raw",
        compatibility_path=tmp_path / "pit_compat.json",
    )
    pit_store.write_snapshot(
        raw_records=[
            PITUniverseRecord(
                symbol="000001.SZ",
                source_list_status=LIST_STATUS_LISTED,
                list_date="20200101",
                observed_at="2026-07-06T00:00:00Z",
                source_run_id="unit-test",
            ),
            PITUniverseRecord(
                symbol="000002.SZ",
                source_list_status=LIST_STATUS_DELISTED,
                list_date="20200101",
                delist_date="20260102",
                observed_at="2026-07-06T00:00:00Z",
                source_run_id="unit-test",
            ),
        ],
        observed_at="2026-07-06T00:00:00Z",
        source_run_id="unit-test",
    )
    reader = MarketDataReader(market="CN", data_root=tmp_path)

    from quant_investor.config import config as runtime_config

    monkeypatch.setattr(runtime_config, "PIT_UNIVERSE_ENABLED", False, raising=False)
    monkeypatch.setattr(
        runtime_config,
        "PIT_UNIVERSE_SOURCE_ROOT",
        str(tmp_path / "parquet" / "cn" / "reference"),
        raising=False,
    )
    assert reader.list_symbols("full_a", as_of="20260103") == ["000001.SZ", "000002.SZ"]

    monkeypatch.setattr(runtime_config, "PIT_UNIVERSE_ENABLED", True, raising=False)
    monkeypatch.setattr(runtime_config, "PIT_UNIVERSE_REQUIRED", False, raising=False)
    assert reader.list_symbols("full_a", as_of="20260103") == ["000001.SZ"]
    assert reader.list_symbols("zz500", as_of="20260103") == []


def test_reader_reuses_snapshot_and_symbol_inventory_for_repeated_runtime_reads(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_parquet_fixture(tmp_path)
    reader = MarketDataReader(market="CN", data_root=tmp_path)
    call_counts = {
        "health_rglob": 0,
        "serving_glob": 0,
        "component_read": 0,
    }
    original_glob = Path.glob
    original_rglob = Path.rglob
    original_read_text = Path.read_text

    def _counting_glob(path: Path, pattern: str, *args, **kwargs):
        if pattern == "symbol=*/bars.parquet":
            call_counts["serving_glob"] += 1
        return original_glob(path, pattern, *args, **kwargs)

    def _counting_rglob(path: Path, pattern: str, *args, **kwargs):
        if pattern == "*.parquet":
            call_counts["health_rglob"] += 1
        return original_rglob(path, pattern, *args, **kwargs)

    def _counting_read_text(path: Path, *args, **kwargs):
        if path.name == "cn_index_components.json":
            call_counts["component_read"] += 1
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "glob", _counting_glob)
    monkeypatch.setattr(Path, "rglob", _counting_rglob)
    monkeypatch.setattr(Path, "read_text", _counting_read_text)

    reader.read_symbol_frames(
        ["000001.SZ", "000002.SZ"],
        columns=["trade_date", "close"],
    )
    assert reader.list_symbols("hs300") == ["000001.SZ"]
    assert reader.list_symbols("zz500") == ["000002.SZ"]

    assert call_counts == {
        "health_rglob": 1,
        "serving_glob": 2,
        "component_read": 1,
    }


def test_reader_batch_reads_symbols_from_canonical_dataset_without_per_symbol_file_reads(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_parquet_fixture(tmp_path)
    reader = MarketDataReader(market="CN", data_root=tmp_path)
    read_parquet_calls = {"count": 0}
    original_read_parquet = pd.read_parquet

    def _counting_read_parquet(*args, **kwargs):
        read_parquet_calls["count"] += 1
        return original_read_parquet(*args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", _counting_read_parquet)

    batch = reader.read_symbol_frames(
        ["000001.SZ", "000002.SZ"],
        columns=["ts_code", "trade_date", "close"],
    )

    assert read_parquet_calls["count"] == 0
    assert sorted(batch) == ["000001.SZ", "000002.SZ"]
    assert batch["000001.SZ"].frame["trade_date"].tolist() == [
        "20260101",
        "20260102",
        "20260103",
    ]
    assert batch["000002.SZ"].frame.to_dict(orient="records") == [
        {"ts_code": "000002.SZ", "trade_date": "20260103", "close": 21.0}
    ]
    assert batch["000001.SZ"].metadata["storage_layer"] == "canonical_batch"
    assert batch["000001.SZ"].metadata["batch_read"] is True


def test_reader_batch_splits_canonical_dataset_without_per_symbol_boolean_scans(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_parquet_fixture(tmp_path)
    reader = MarketDataReader(market="CN", data_root=tmp_path)
    series_eq_calls = {"count": 0}
    original_series_eq = pd.Series.eq

    def _counting_series_eq(self, other, *args, **kwargs):
        series_eq_calls["count"] += 1
        return original_series_eq(self, other, *args, **kwargs)

    monkeypatch.setattr(pd.Series, "eq", _counting_series_eq)

    batch = reader.read_symbol_frames(
        ["000001.SZ", "000002.SZ"],
        columns=["ts_code", "trade_date", "close"],
    )

    assert sorted(batch) == ["000001.SZ", "000002.SZ"]
    assert series_eq_calls["count"] == 0


def test_reader_batch_uses_dataset_symbol_filter_without_pandas_refilter(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_parquet_fixture(tmp_path)
    reader = MarketDataReader(market="CN", data_root=tmp_path)
    series_isin_calls = {"count": 0}
    original_series_isin = pd.Series.isin

    def _counting_series_isin(self, values, *args, **kwargs):
        series_isin_calls["count"] += 1
        return original_series_isin(self, values, *args, **kwargs)

    monkeypatch.setattr(pd.Series, "isin", _counting_series_isin)

    batch = reader.read_symbol_frames(
        ["000001.SZ", "000002.SZ"],
        columns=["ts_code", "trade_date", "close"],
    )

    assert sorted(batch) == ["000001.SZ", "000002.SZ"]
    assert series_isin_calls["count"] == 0


def test_reader_batch_split_reuses_group_frames_without_extra_copies(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_parquet_fixture(tmp_path)
    reader = MarketDataReader(market="CN", data_root=tmp_path)
    batch_frame = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "symbol": "000001.SZ", "trade_date": "20260101", "close": 10.0},
            {"ts_code": "000001.SZ", "symbol": "000001.SZ", "trade_date": "20260102", "close": 11.0},
            {"ts_code": "000002.SZ", "symbol": "000002.SZ", "trade_date": "20260103", "close": 21.0},
        ]
    )
    copy_calls = {"count": 0}
    original_copy = pd.DataFrame.copy

    monkeypatch.setattr(reader, "_read_dataset", lambda *args, **kwargs: batch_frame)

    def _counting_copy(self, *args, **kwargs):
        copy_calls["count"] += 1
        return original_copy(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "copy", _counting_copy)

    batch = reader.read_symbol_frames(
        ["000001.SZ", "000002.SZ"],
        columns=["ts_code", "trade_date", "close"],
    )

    assert batch["000001.SZ"].frame.to_dict(orient="records") == [
        {"ts_code": "000001.SZ", "trade_date": "20260101", "close": 10.0},
        {"ts_code": "000001.SZ", "trade_date": "20260102", "close": 11.0},
    ]
    assert batch["000002.SZ"].frame.to_dict(orient="records") == [
        {"ts_code": "000002.SZ", "trade_date": "20260103", "close": 21.0}
    ]
    assert copy_calls["count"] <= 5


def test_reader_batch_groups_by_ts_code_without_deriving_symbol_column_copy(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_parquet_fixture(tmp_path)
    reader = MarketDataReader(market="CN", data_root=tmp_path)
    copy_calls = {"count": 0}
    original_copy = pd.DataFrame.copy

    def _counting_copy(self, *args, **kwargs):
        copy_calls["count"] += 1
        return original_copy(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "copy", _counting_copy)

    batch = reader.read_symbol_frames(
        ["000001.SZ", "000002.SZ"],
        columns=["ts_code", "trade_date", "close"],
    )

    assert sorted(batch) == ["000001.SZ", "000002.SZ"]
    assert "symbol" not in batch["000001.SZ"].frame.columns
    assert "symbol" not in batch["000002.SZ"].frame.columns
    assert copy_calls["count"] <= 7


def test_reader_batch_normalizes_symbol_lists_once_per_boundary(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import quant_investor.market.market_data_reader as reader_module

    _write_parquet_fixture(tmp_path)
    normalize_calls = {"count": 0}
    original_normalize = reader_module._normalize_symbol

    def _counting_normalize(value):
        normalize_calls["count"] += 1
        return original_normalize(value)

    monkeypatch.setattr(reader_module, "_normalize_symbol", _counting_normalize)
    reader = reader_module.MarketDataReader(market="CN", data_root=tmp_path)

    batch = reader.read_symbol_frames(
        ["000001.SZ", "000002.SZ"],
        columns=["ts_code", "trade_date", "close"],
    )

    assert sorted(batch) == ["000001.SZ", "000002.SZ"]
    assert normalize_calls["count"] <= 6


def test_reader_batch_skips_serving_inventory_when_dataset_covers_requested_symbols(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_parquet_fixture(tmp_path)
    reader = MarketDataReader(market="CN", data_root=tmp_path)
    serving_inventory_calls = {"count": 0}
    original_serving_symbols = reader._serving_symbols

    def _counting_serving_symbols(snapshot):
        serving_inventory_calls["count"] += 1
        return original_serving_symbols(snapshot)

    monkeypatch.setattr(reader, "_serving_symbols", _counting_serving_symbols)

    batch = reader.read_symbol_frames(
        ["000001.SZ", "000002.SZ"],
        columns=["ts_code", "trade_date", "close"],
    )

    assert sorted(batch) == ["000001.SZ", "000002.SZ"]
    assert serving_inventory_calls["count"] == 0


def test_reader_dataset_fallback_keeps_cross_section_as_of_exact(monkeypatch, tmp_path: Path) -> None:
    _write_parquet_fixture(tmp_path)

    import pyarrow.dataset as ds

    monkeypatch.setattr(ds, "dataset", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("force fallback")))
    reader = MarketDataReader(market="CN", data_root=tmp_path)

    cross_section = reader.read_cross_section(
        "20260103",
        columns=["ts_code", "trade_date", "close"],
    )

    assert len(cross_section) == 2
    assert set(cross_section["symbol"]) == {"000001.SZ", "000002.SZ"}
    assert set(cross_section["trade_date"]) == {"20260103"}


def test_strict_parquet_reader_fails_closed_and_never_falls_back_to_csv(tmp_path: Path) -> None:
    csv_dir = tmp_path / "cn_market_full" / "hs300"
    csv_dir.mkdir(parents=True)
    (csv_dir / "000001.SZ.csv").write_text(
        "trade_date,close\n20260103,99\n",
        encoding="utf-8",
    )

    reader = MarketDataReader(market="CN", data_root=tmp_path)
    with pytest.raises(MarketDataUnavailableError):
        reader.list_symbols("full_a")

    _write_parquet_fixture(tmp_path, status="BLOCKED")
    blocked_reader = MarketDataReader(market="CN", data_root=tmp_path)
    with pytest.raises(MarketDataUnavailableError):
        blocked_reader.read_symbol_frame("000001.SZ")


def test_market_data_store_validate_latest_rejects_incomplete_snapshot(tmp_path: Path) -> None:
    _write_parquet_fixture(tmp_path)
    store = MarketDataStore(market="CN", data_root=tmp_path)
    assert store.validate_latest()["status"] == "passed"

    (tmp_path / "parquet" / "cn" / "_snapshots" / "snap-001.json").unlink()
    failed = store.validate_latest()
    assert failed["status"] == "failed"
    assert any("manifest" in blocker for blocker in failed["blockers"])


def test_market_backtest_loads_cn_frame_from_parquet_serving(tmp_path: Path) -> None:
    _write_parquet_fixture(tmp_path)

    frame = _load_market_frame(
        "CN",
        ["hs300"],
        data_dir=str(tmp_path),
        sample_size=1,
    )

    assert frame["symbol"].unique().tolist() == ["000001.SZ"]
    assert set(["date", "close", "forward_ret_1d", "factor_score"]).issubset(frame.columns)
    assert len(frame) == 2
