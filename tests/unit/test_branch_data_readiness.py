from __future__ import annotations

import json

import pandas as pd

from quant_investor.market.branch_readiness import (
    STATUS_BLOCK,
    STATUS_PASS,
    assess_branch_data_readiness,
    assess_quant_readiness,
    write_branch_readiness_report,
)
from quant_investor.market.intelligence_mart import build_intelligence_daily, write_intelligence_mart
from quant_investor.market.macro_mart import write_macro_mart


def _price_frame(symbol: str = "000001.SZ") -> pd.DataFrame:
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


def _write_fundamental_daily(root, symbols=("000001.SZ",)):
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for symbol in symbols:
        rows.append(
            {
                "ts_code": symbol,
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
        )
    pd.DataFrame(rows).to_csv(root / "fundamental_daily.csv", index=False)
    (root / "latest_manifest.json").write_text(
        json.dumps({"provider_status": "tushare_primary", "source_priority": "tushare_primary"}),
        encoding="utf-8",
    )


def test_quant_missing_required_amount_hard_blocks():
    frame = _price_frame().drop(columns=["amount"])

    readiness = assess_quant_readiness(frames={"000001.SZ": frame}, symbols=["000001.SZ"], as_of="20240510")

    assert readiness.status == STATUS_BLOCK
    assert "amount" in readiness.missing_fields
    assert readiness.affected_symbols == ["000001.SZ"]


def test_four_branch_readiness_blocks_only_symbols_missing_enabled_branch_data(tmp_path):
    fundamental_root = tmp_path / "cn_fundamental"
    intelligence_root = tmp_path / "cn_intelligence"
    macro_root = tmp_path / "cn_macro"
    _write_fundamental_daily(fundamental_root, symbols=("000001.SZ",))
    frames = {
        "000001.SZ": _price_frame("000001.SZ"),
        "000002.SZ": _price_frame("000002.SZ"),
    }
    intelligence_daily = build_intelligence_daily(frames)
    write_intelligence_mart(intelligence_daily, data_root=intelligence_root, raw_snapshot_root=tmp_path / "snapshots" / "intelligence")
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

    report = assess_branch_data_readiness(
        frames=frames,
        candidate_symbols=["000001.SZ", "000002.SZ"],
        as_of="20240510",
        fundamental_root=fundamental_root,
        intelligence_root=intelligence_root,
        macro_root=macro_root,
        run_id="fixture",
    )

    assert report.readiness["quant"].status == STATUS_PASS
    assert report.readiness["fundamental"].status == STATUS_BLOCK
    assert report.readiness["intelligence"].status == STATUS_PASS
    assert report.readiness["macro"].status == STATUS_PASS
    assert report.blocked_symbols == ["000002.SZ"]
    assert report.investable_universe == ["000001.SZ"]
    assert report.branch_data["fundamentals"]["000001.SZ"]["forecast_revision"] == 0.05


def test_macro_missing_blocks_four_branch_fusion(tmp_path):
    fundamental_root = tmp_path / "cn_fundamental"
    intelligence_root = tmp_path / "cn_intelligence"
    macro_root = tmp_path / "missing_macro"
    _write_fundamental_daily(fundamental_root, symbols=("000001.SZ",))
    frames = {"000001.SZ": _price_frame("000001.SZ")}
    intelligence_daily = build_intelligence_daily(frames)
    write_intelligence_mart(intelligence_daily, data_root=intelligence_root, raw_snapshot_root=tmp_path / "snapshots" / "intelligence")

    report = assess_branch_data_readiness(
        frames=frames,
        candidate_symbols=["000001.SZ"],
        as_of="20240510",
        fundamental_root=fundamental_root,
        intelligence_root=intelligence_root,
        macro_root=macro_root,
        run_id="fixture",
    )

    assert report.readiness["macro"].status == STATUS_BLOCK
    assert "000001.SZ" in report.blocked_symbols
    assert report.investable_universe == []


def test_branch_readiness_writes_standard_artifacts(tmp_path):
    report = assess_branch_data_readiness(
        frames={"000001.SZ": _price_frame("000001.SZ")},
        candidate_symbols=["000001.SZ"],
        as_of="20240510",
        fundamental_root=tmp_path / "missing_fundamental",
        intelligence_root=tmp_path / "missing_intelligence",
        macro_root=tmp_path / "missing_macro",
        run_id="fixture",
    )

    artifacts = write_branch_readiness_report(report, output_dir=tmp_path / "reports")

    assert set(artifacts) == {"json", "md", "csv"}
    assert json.loads((tmp_path / "reports" / "fixture.json").read_text())["readiness"]["quant"]["status"] == STATUS_PASS
