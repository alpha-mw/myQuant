from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_investor.market.market_data_reader import MarketDataReader
from tests.unit.test_market_data_reader_parquet import (
    _write_legacy_parquet_fixture,
    _write_v4_parquet_fixture,
)


def _rewrite_v4_fixture_as_v3(paths: dict[str, Path]) -> None:
    latest = json.loads(paths["latest"].read_text(encoding="utf-8"))
    latest["coverage"]["coverage_schema_version"] = "cn-full-a-coverage.v3"
    paths["latest"].write_text(json.dumps(latest), encoding="utf-8")

    manifest_path = Path(latest["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["coverage"]["coverage_schema_version"] = (
        "cn-full-a-coverage.v3"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _write_us_legacy_fixture(root: Path) -> None:
    table_root = root / "parquet" / "us" / "bars" / "year=2026"
    serving_root = (
        root / "parquet_serving" / "us" / "bars" / "symbol=AAPL"
    )
    snapshot_root = root / "parquet" / "us" / "_snapshots"
    table_root.mkdir(parents=True)
    serving_root.mkdir(parents=True)
    snapshot_root.mkdir(parents=True)

    frame = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "trade_date": "20260103",
                "close": 250.0,
            }
        ]
    )
    frame.to_parquet(table_root / "part.parquet", index=False)
    frame.to_parquet(serving_root / "bars.parquet", index=False)

    manifest_path = snapshot_root / "us-snap-001.json"
    manifest_path.write_text(
        json.dumps({"snapshot_id": "us-snap-001"}),
        encoding="utf-8",
    )
    latest = {
        "status": "OK",
        "snapshot_id": "us-snap-001",
        "latest_complete_trade_date": "20260103",
        "latest_trade_date": "20260103",
        "table_root": str(root / "parquet" / "us" / "bars"),
        "derived_serving_root": str(
            root / "parquet_serving" / "us" / "bars"
        ),
        "manifest_path": str(manifest_path),
        "coverage": {"row_count": 1, "symbol_count": 1},
        "blockers": [],
    }
    (root / "parquet" / "us" / "_latest.json").write_text(
        json.dumps(latest),
        encoding="utf-8",
    )


def test_cn_strict_rejects_unversioned_legacy_coverage(tmp_path: Path) -> None:
    _write_legacy_parquet_fixture(tmp_path)

    gate = MarketDataReader(
        market="CN",
        data_root=tmp_path,
        mode_policy="strict",
    ).clean_snapshot_gate()

    assert gate["healthy"] is False
    assert gate["blockers"] == [
        "cn_strict_coverage_schema_v4_required:missing"
    ]


def test_cn_strict_rejects_v3_coverage(tmp_path: Path) -> None:
    paths = _write_v4_parquet_fixture(tmp_path)
    _rewrite_v4_fixture_as_v3(paths)

    gate = MarketDataReader(
        market="CN",
        data_root=tmp_path,
        mode_policy="strict",
    ).clean_snapshot_gate()

    assert gate["healthy"] is False
    assert "cn_strict_coverage_schema_v4_required:cn-full-a-coverage.v3" in (
        gate["blockers"]
    )


def test_cn_strict_accepts_exact_v4_snapshot(tmp_path: Path) -> None:
    _write_v4_parquet_fixture(tmp_path)

    gate = MarketDataReader(
        market="CN",
        data_root=tmp_path,
        mode_policy="strict",
    ).clean_snapshot_gate()

    assert gate["healthy"] is True
    assert gate["blockers"] == []


def test_us_strict_legacy_snapshot_is_unchanged(tmp_path: Path) -> None:
    _write_us_legacy_fixture(tmp_path)

    gate = MarketDataReader(
        market="US",
        data_root=tmp_path,
        mode_policy="strict",
    ).clean_snapshot_gate()

    assert gate["healthy"] is True
    assert gate["blockers"] == []
