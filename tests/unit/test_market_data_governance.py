from __future__ import annotations

import importlib
import json

import pandas as pd
import pytest

from quant_investor.market.data_governance import run_data_governance
from quant_investor.market.macro_mart import write_macro_mart


def _daily_frame(symbol: str = "000001.SZ") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": [symbol, symbol],
            "trade_date": ["20240509", "20240510"],
            "open": [10.0, 10.2],
            "high": [10.3, 10.5],
            "low": [9.9, 10.0],
            "close": [10.1, 10.4],
            "volume": [1000.0, 1200.0],
            "amount": [10_000.0, 12_500.0],
        }
    )


def _write_parquet_market_data(root):
    data_root = root / "data"
    parquet_root = data_root / "parquet" / "cn"
    bars_root = parquet_root / "bars"
    serving_root = data_root / "parquet_serving" / "cn" / "bars"
    manifest_path = parquet_root / "_snapshots" / "fixture.json"
    frame = _daily_frame()
    (bars_root / "year=2024").mkdir(parents=True)
    (serving_root / "symbol=000001.SZ").mkdir(parents=True)
    frame.to_parquet(bars_root / "year=2024" / "part.parquet", index=False)
    frame.to_parquet(serving_root / "symbol=000001.SZ" / "bars.parquet", index=False)
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({"snapshot_id": "fixture"}), encoding="utf-8")
    (parquet_root / "_latest.json").write_text(
        json.dumps(
            {
                "status": "OK",
                "snapshot_id": "fixture",
                "latest_complete_trade_date": "20240510",
                "latest_trade_date": "20240510",
                "table_root": str(bars_root),
                "derived_serving_root": str(serving_root),
                "manifest_path": str(manifest_path),
            }
        ),
        encoding="utf-8",
    )
    return data_root


def _write_fundamental(root):
    root.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20240510",
                "availability_date": "2024-04-30",
                "source": "tushare_fina_indicator;forecast",
                "source_priority": "tushare_primary",
                "fin_roe": 0.13,
                "fin_roa": 0.06,
                "fin_debt_to_assets": 0.42,
                "fin_net_profit_yoy": 0.18,
                "fin_ocf_to_profit": 1.12,
                "fin_fcf_to_profit": 0.88,
                "fcf_to_price": 0.04,
                "forecast_revision": 0.05,
            }
        ]
    ).to_parquet(root / "part.parquet", index=False)
    (root / "latest_manifest.json").write_text(
        json.dumps(
            {
                "provider_status": "tushare_primary",
                "source_priority": "tushare_primary",
                "storage_backend": "parquet_canonical",
            }
        ),
        encoding="utf-8",
    )


def test_data_governance_default_is_local_read_only(tmp_path):
    data_root = _write_parquet_market_data(tmp_path)
    fundamental_root = tmp_path / "cn_fundamental"
    macro_root = tmp_path / "cn_macro"
    _write_fundamental(fundamental_root)
    write_macro_mart(
        {
            "trade_date": "20240510",
            "macro_score": 0.2,
            "liquidity_score": 0.4,
            "volatility_percentile": 45.0,
            "policy_signal": "neutral",
            "source": "tushare_macro",
            "source_priority": "tushare_primary",
        },
        data_root=macro_root,
        raw_snapshot_root=tmp_path / "snapshots" / "macro",
    )

    result = run_data_governance(
        market="CN",
        categories=["full_a"],
        as_of="20240510",
        data_dir=data_root,
        fundamental_root=fundamental_root,
        macro_root=macro_root,
        output_dir=tmp_path / "reports",
    )

    assert result["local_read_only"] is True
    assert result["allow_live"] is False
    assert result["reports"][0]["readiness"]["macro"]["status"] == "pass"
    assert (tmp_path / "reports").joinpath(result["artifacts"]["full_a"]["json"].split("/")[-1]).exists()


def test_data_governance_allow_live_uses_explicit_maintenance_path(tmp_path, monkeypatch):
    data_root = _write_parquet_market_data(tmp_path)
    calls = {"fundamental": 0, "macro": 0}

    def _fake_fundamental(**kwargs):
        calls["fundamental"] += 1
        assert kwargs["allow_live"] is True
        assert kwargs["universes"] == "full_a"
        return {}

    def _fake_macro(**kwargs):
        calls["macro"] += 1
        assert kwargs["allow_live"] is True
        return {}

    monkeypatch.setattr("quant_investor.market.fundamental_mart.run_cn_fundamental_maintenance", _fake_fundamental)
    monkeypatch.setattr("quant_investor.market.macro_mart.run_cn_macro_maintenance", _fake_macro)

    result = run_data_governance(
        market="CN",
        categories=["full_a"],
        as_of="20240510",
        allow_live=True,
        data_dir=data_root,
        fundamental_root=tmp_path / "fundamental",
        macro_root=tmp_path / "macro",
        output_dir=tmp_path / "reports",
    )

    assert calls == {"fundamental": 1, "macro": 1}
    assert result["local_read_only"] is False


def test_data_governance_rejects_retired_root_argument(tmp_path):
    with pytest.raises(TypeError):
        run_data_governance(
            intelligence_root=tmp_path / "retired",  # type: ignore[call-arg]
        )


def test_retired_market_mart_module_is_physically_absent():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("quant_investor.market.intelligence_mart")
