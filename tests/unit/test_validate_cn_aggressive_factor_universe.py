from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from tests.fixtures.strict_cn_snapshot import coverage_v4, v4_snapshot_paths

ROOT = Path(__file__).resolve().parents[2]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "validate_cn_aggressive_factor_universe",
        ROOT / "scripts" / "validate_cn_aggressive_factor_universe.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_output_is_research_only():
    module = _load_script()

    assert module.DEFAULT_OUTPUT == Path(
        "results/research/CN/aggressive_tech_manufacturing/"
        "20260618_broad_factor_validation"
    )
    assert "strategy_records" not in module.DEFAULT_OUTPUT.parts


def _write_snapshot(
    tmp_path: Path,
    *,
    table_root: Path | None = None,
    status: str = "OK",
) -> tuple[Path, Path]:
    data_root = tmp_path / "data"
    market_root = data_root / "parquet" / "cn"
    active_root, serving_root, manifest_path = v4_snapshot_paths(data_root, "active")
    latest_path = market_root / "_latest.json"
    selected_table_root = table_root or active_root

    selected_table_root.mkdir(parents=True, exist_ok=True)
    serving_symbol_root = serving_root / "symbol=000001.SZ"
    serving_symbol_root.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": ["20260715"],
            "open": [9.8],
            "high": [10.2],
            "low": [9.7],
            "close": [10.0],
            "adj_close": [10.0],
            "amount": [1_000_000.0],
            "vol": [100_000.0],
        }
    )
    frame.to_parquet(selected_table_root / "part.parquet", index=False)
    frame.to_parquet(serving_symbol_root / "bars.parquet", index=False)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    coverage = coverage_v4(data_root, ["000001.SZ"], trade_date="20260715")
    manifest_path.write_text(
        json.dumps({"snapshot_id": "active", "coverage": coverage}),
        encoding="utf-8",
    )
    latest_path.write_text(
        json.dumps(
            {
                "status": status,
                "snapshot_id": "active",
                "latest_complete_trade_date": "20260715",
                "latest_trade_date": "20260715",
                "table_root": str(selected_table_root.resolve()),
                "derived_serving_root": str(serving_root.resolve()),
                "manifest_path": str(manifest_path.resolve()),
                "coverage": coverage,
                "blockers": [],
            }
        ),
        encoding="utf-8",
    )
    return latest_path, active_root


def test_resolves_table_root_from_active_pointer_not_legacy_fixed_root(tmp_path):
    module = _load_script()
    latest_path, active_root = _write_snapshot(tmp_path)
    legacy_root = latest_path.parent / "bars"
    legacy_root.mkdir()
    (legacy_root / "legacy-only.txt").write_text("stale", encoding="utf-8")

    bars_root, payload, pointer_sha256 = module._resolve_active_bars_root(latest_path)

    assert bars_root == active_root.resolve()
    assert payload["snapshot_id"] == "active"
    assert len(pointer_sha256) == 64
    assert bars_root != legacy_root.resolve()


def test_base_frame_reads_pointer_bound_table_root(tmp_path, monkeypatch):
    module = _load_script()
    latest_path, active_root = _write_snapshot(tmp_path)
    stock_basic_path = tmp_path / "stock-basic.parquet"
    daily_basic_path = tmp_path / "missing-daily-basic.parquet"
    pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "name": ["平安银行"],
            "industry": ["银行"],
            "market": ["主板"],
            "list_date": ["19910403"],
        }
    ).to_parquet(stock_basic_path, index=False)
    monkeypatch.setattr(module, "LATEST_PATH", latest_path)
    monkeypatch.setattr(module, "STOCK_BASIC_PATH", stock_basic_path)
    monkeypatch.setattr(module, "DAILY_BASIC_PATH", daily_basic_path)

    frame, lineage = module._load_base_frame(module.ValidationConfig(warmup_date="20260701"))

    assert frame["symbol"].tolist() == ["000001.SZ"]
    assert lineage["bars_root"] == str(active_root.resolve())
    assert lineage["latest_pointer_path"] == str(latest_path)
    assert len(lineage["latest_pointer_sha256"]) == 64


def test_missing_table_root_does_not_fall_back_to_legacy_root(tmp_path):
    module = _load_script()
    latest_path, _ = _write_snapshot(tmp_path)
    legacy_root = latest_path.parent / "bars"
    legacy_root.mkdir()
    pd.DataFrame({"ts_code": ["000001.SZ"]}).to_parquet(
        legacy_root / "part.parquet",
        index=False,
    )
    payload = json.loads(latest_path.read_text(encoding="utf-8"))
    payload.pop("table_root")
    latest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(module.ActiveMarketSnapshotError, match="table_root is missing"):
        module._resolve_active_bars_root(latest_path)


def test_rejects_active_table_root_outside_cn_canonical_root(tmp_path):
    module = _load_script()
    outside_root = tmp_path / "outside-bars"
    latest_path, _ = _write_snapshot(tmp_path, table_root=outside_root)

    with pytest.raises(
        module.ActiveMarketSnapshotError,
        match="absolute path escape rejected|escapes CN canonical root",
    ):
        module._resolve_active_bars_root(latest_path)


def test_rejects_symlinked_active_table_root(tmp_path):
    module = _load_script()
    outside_root = tmp_path / "outside-bars"
    outside_root.mkdir()
    pd.DataFrame({"ts_code": ["000001.SZ"]}).to_parquet(
        outside_root / "part.parquet",
        index=False,
    )
    latest_path, active_root = _write_snapshot(tmp_path)
    for path in active_root.iterdir():
        path.unlink()
    active_root.rmdir()
    active_root.symlink_to(outside_root, target_is_directory=True)

    with pytest.raises(module.ActiveMarketSnapshotError, match="symlink"):
        module._resolve_active_bars_root(latest_path)


def test_rejects_blocked_active_pointer(tmp_path):
    module = _load_script()
    latest_path, _ = _write_snapshot(tmp_path, status="PARTIAL")

    with pytest.raises(module.ActiveMarketSnapshotError, match="snapshot is blocked"):
        module._resolve_active_bars_root(latest_path)


def test_rejects_symlinked_active_pointer(tmp_path):
    module = _load_script()
    latest_path, _ = _write_snapshot(tmp_path)
    real_pointer = latest_path.with_name("_latest.real.json")
    latest_path.replace(real_pointer)
    latest_path.symlink_to(real_pointer)

    with pytest.raises(
        module.ActiveMarketSnapshotError,
        match="must not be a symlink",
    ):
        module._resolve_active_bars_root(latest_path)
