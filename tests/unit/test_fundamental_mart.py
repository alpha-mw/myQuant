from __future__ import annotations

import json

import pandas as pd

import quant_investor.market.fundamental_mart as fundamental_mart
from quant_investor.factors.pit_fundamentals import (
    build_fundamental_metric_matrices,
    load_fundamental_pit_series,
)
from quant_investor.market.fundamental_mart import (
    DERIVED_DAILY_FIELDS,
    _fetch_tushare_tables,
    _resolve_symbols_from_daily_root,
    run_cn_fundamental_maintenance,
    write_fundamental_mart,
)


def _raw_tables() -> dict[str, pd.DataFrame]:
    symbols = ["000001.SZ", "000002.SZ", "000003.SZ"]
    fina_rows = []
    income_rows = []
    balance_rows = []
    cashflow_rows = []
    daily_rows = []
    forecast_rows = []
    for idx, symbol in enumerate(symbols, start=1):
        sector = ["bank", "industrial", "healthcare"][idx - 1]
        for end_date, ann_date, profit, ocf, capex, roe in (
            ("20221231", "20230428", 80.0 + idx, 100.0 + idx, 10.0, 8.0 + idx),
            ("20231231", "20240430", 100.0 + idx, 130.0 + idx, 20.0, 12.0 + idx),
        ):
            fina_rows.append(
                {
                    "ts_code": symbol,
                    "end_date": end_date,
                    "ann_date": ann_date,
                    "f_ann_date": ann_date,
                    "roe_dt": roe,
                    "roa": 5.0 + idx,
                    "debt_to_assets": 45.0 + idx,
                    "netprofit_yoy": "",
                    "ocf_to_profit": "",
                }
            )
            income_rows.append(
                {
                    "ts_code": symbol,
                    "end_date": end_date,
                    "ann_date": ann_date,
                    "f_ann_date": ann_date,
                    "n_income_attr_p": profit,
                }
            )
            balance_rows.append(
                {
                    "ts_code": symbol,
                    "end_date": end_date,
                    "ann_date": ann_date,
                    "f_ann_date": ann_date,
                    "total_liab": 400.0 + idx,
                    "total_assets": 1000.0 + idx,
                }
            )
            cashflow_rows.append(
                {
                    "ts_code": symbol,
                    "end_date": end_date,
                    "ann_date": ann_date,
                    "f_ann_date": ann_date,
                    "n_cashflow_act": ocf,
                    "c_pay_acq_const_fiolta": capex,
                }
            )
        for trade_date in ("20240429", "20240430", "20240502", "20240510"):
            daily_rows.append(
                {
                    "ts_code": symbol,
                    "trade_date": trade_date,
                    "total_mv": 100000.0 * idx,
                    "sector": sector,
                }
            )
        forecast_rows.append(
            {
                "ts_code": symbol,
                "ann_date": "20240429",
                "end_date": "20240630",
                "type": "预增",
                "p_change_min": 5.0 + idx,
                "p_change_max": 15.0 + idx,
                "summary": "fixture forecast",
                "change_reason": "fixture",
            }
        )
    fina_rows.append(
        {
            "ts_code": "000001.SZ",
            "end_date": "20231231",
            "ann_date": "20240510",
            "f_ann_date": "20240510",
            "roe_dt": 20.0,
            "roa": 7.0,
            "debt_to_assets": 40.0,
        }
    )
    fina_rows.append(
        {
            "ts_code": "000004.SZ",
            "end_date": "20231231",
            "roe_dt": 10.0,
        }
    )
    return {
        "fina_indicator": pd.DataFrame(fina_rows),
        "income": pd.DataFrame(income_rows),
        "balancesheet": pd.DataFrame(balance_rows),
        "cashflow": pd.DataFrame(cashflow_rows),
        "daily_basic": pd.DataFrame(daily_rows),
        "forecast": pd.DataFrame(forecast_rows),
    }


def test_fundamental_mart_pit_join_readiness_and_quarantine(tmp_path):
    artifacts, readiness = write_fundamental_mart(
        _raw_tables(),
        data_root=tmp_path / "clean" / "cn_fundamental",
        raw_snapshot_root=tmp_path / "snapshots" / "fundamental",
        reports_root=tmp_path / "reports" / "fundamental_readiness",
        run_id="fixture",
    )

    daily = pd.read_csv(artifacts.fundamental_daily_path)
    symbol_daily = daily[daily["ts_code"] == "000001.SZ"].set_index("trade_date")
    assert symbol_daily.loc["2024-04-29", "fin_roe"] == 0.09
    assert symbol_daily.loc["2024-04-30", "fin_roe"] == 0.13
    assert symbol_daily.loc["2024-05-02", "fin_roe"] == 0.13
    assert symbol_daily.loc["2024-05-10", "fin_roe"] == 0.20
    assert symbol_daily.loc["2024-04-30", "fin_net_profit_yoy"] > 0.0
    assert symbol_daily.loc["2024-04-30", "fin_fcf_to_profit"] > 0.0
    assert symbol_daily.loc["2024-04-30", "fcf_to_price"] > 0.0
    assert symbol_daily.loc["2024-04-30", "forecast_revision"] == 0.11
    assert readiness["gate2_passed"] is True
    assert readiness["coverage_rate"] >= 0.60
    quarantine = pd.read_csv(artifacts.quarantine_path)
    assert "missing_ts_code_end_date_or_announcement_date" in set(quarantine["quarantine_reason"])
    assert artifacts.readiness_json_path.exists()
    assert json.loads(artifacts.readiness_json_path.read_text())["gate2_passed"] is True


def test_legacy_fetched_at_fallback_disabled_for_production(tmp_path):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "period": "20231231",
                "metric_name": "operating_cashflow",
                "value": 100.0,
                "fetched_at": "2024-05-10",
            }
        ]
    ).to_csv(metadata_dir / "fundamental_series.csv", index=False)

    production = load_fundamental_pit_series(
        metadata_dir=metadata_dir,
        mart_root=tmp_path / "missing_mart",
        allow_legacy_fallback=False,
    )
    diagnostic = load_fundamental_pit_series(
        metadata_dir=metadata_dir,
        mart_root=tmp_path / "missing_mart",
        allow_legacy_fallback=True,
    )

    assert production.empty
    assert not diagnostic.empty
    assert diagnostic["source"].str.contains("fetched_at_fallback").any()


def test_metric_matrices_read_canonical_mart_without_legacy(tmp_path):
    artifacts, _readiness = write_fundamental_mart(
        _raw_tables(),
        data_root=tmp_path / "clean" / "cn_fundamental",
        raw_snapshot_root=tmp_path / "snapshots" / "fundamental",
        reports_root=tmp_path / "reports" / "fundamental_readiness",
        run_id="fixture",
    )
    dates = pd.to_datetime(["2024-04-29", "2024-04-30"])
    matrices, diagnostics = build_fundamental_metric_matrices(
        dates,
        ["000001.SZ"],
        metrics=DERIVED_DAILY_FIELDS,
        mart_root=artifacts.data_root,
        allow_legacy_fallback=False,
    )

    assert diagnostics["legacy_fallback_allowed"] is False
    assert matrices["fin_roe"].loc[pd.Timestamp("2024-04-29"), "000001.SZ"] == 0.09
    assert matrices["fin_roe"].loc[pd.Timestamp("2024-04-30"), "000001.SZ"] == 0.13
    assert matrices["fcf_to_price"].loc[pd.Timestamp("2024-04-30"), "000001.SZ"] > 0.0


def test_daily_basic_uses_local_stock_list_sector(tmp_path, monkeypatch):
    metadata_root = tmp_path / "metadata"
    metadata_root.mkdir()
    pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "industry": "银行"},
            {"ts_code": "000002.SZ", "industry": "全国地产"},
            {"ts_code": "000003.SZ", "industry": "医疗服务"},
        ]
    ).to_csv(metadata_root / "stock_list.csv", index=False)
    monkeypatch.setattr(fundamental_mart, "DEFAULT_METADATA_ROOT", metadata_root)
    tables = _raw_tables()
    tables["daily_basic"] = tables["daily_basic"].drop(columns=["sector"])

    artifacts, _readiness = write_fundamental_mart(
        tables,
        data_root=tmp_path / "clean" / "cn_fundamental",
        raw_snapshot_root=tmp_path / "snapshots" / "fundamental",
        reports_root=tmp_path / "reports" / "fundamental_readiness",
        run_id="fixture",
    )

    daily = pd.read_csv(artifacts.fundamental_daily_path)
    sector_by_symbol = daily.groupby("ts_code")["sector"].first().to_dict()
    assert sector_by_symbol["000001.SZ"] == "银行"
    assert sector_by_symbol["000002.SZ"] == "全国地产"


def test_fundamental_maintenance_offline_input_writes_expected_artifacts(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for table, frame in _raw_tables().items():
        frame.to_csv(raw_dir / f"{table}.csv", index=False)

    result = run_cn_fundamental_maintenance(
        market="CN",
        universes="hs300,zz500,zz1000",
        years=5,
        as_of="20240510",
        raw_input_dir=raw_dir,
        data_root=tmp_path / "clean" / "cn_fundamental",
        raw_snapshot_root=tmp_path / "snapshots" / "fundamental",
        reports_root=tmp_path / "reports" / "fundamental_readiness",
    )

    assert result["provider_status"] == "offline_input"
    assert result["readiness"]["gate2_passed"] is True
    assert result["readiness"]["raw_row_counts"]["fina_indicator"] > 0
    assert (tmp_path / "clean" / "cn_fundamental" / "latest_manifest.json").exists()
    manifest = json.loads((tmp_path / "clean" / "cn_fundamental" / "latest_manifest.json").read_text())
    assert manifest["raw_row_counts"]["daily_basic"] > 0


def test_full_a_universe_resolves_all_physical_daily_directories(tmp_path):
    daily_root = tmp_path / "cn_daily"
    for directory, symbol in {
        "hs300": "000001.SZ",
        "zz500": "000002.SZ",
        "zz1000": "000003.SZ",
        "other": "000004.SZ",
    }.items():
        target = daily_root / directory
        target.mkdir(parents=True)
        pd.DataFrame({"trade_date": ["20240510"], "close": [10.0]}).to_csv(target / f"{symbol}.csv", index=False)

    symbols = _resolve_symbols_from_daily_root(daily_root, ["full_a"])

    assert symbols == ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"]


def test_live_fetch_records_partial_provider_errors(monkeypatch):
    class _FakePro:
        def __init__(self):
            self.calls = []

        def fina_indicator(self, ts_code, start_date="", end_date="", fields=""):
            self.calls.append(("fina_indicator", ts_code, start_date, end_date))
            if ts_code == "000002.SZ":
                raise RuntimeError("quota limited")
            return pd.DataFrame(
                [
                    {
                        "ts_code": ts_code,
                        "end_date": "20231231",
                        "ann_date": "20240430",
                        "roe": 10.0,
                    }
                ]
            )

        def income(self, **kwargs):
            self.calls.append(("income", kwargs.get("ts_code"), kwargs.get("start_date"), kwargs.get("end_date")))
            return pd.DataFrame()

        def balancesheet(self, **kwargs):
            self.calls.append(("balancesheet", kwargs.get("ts_code"), kwargs.get("start_date"), kwargs.get("end_date")))
            return pd.DataFrame()

        def cashflow(self, **kwargs):
            self.calls.append(("cashflow", kwargs.get("ts_code"), kwargs.get("start_date"), kwargs.get("end_date")))
            return pd.DataFrame()

        def daily_basic(self, **kwargs):
            self.calls.append(("daily_basic", kwargs.get("ts_code"), kwargs.get("start_date"), kwargs.get("end_date")))
            return pd.DataFrame()

        def forecast(self, **kwargs):
            self.calls.append(("forecast", kwargs.get("ts_code"), kwargs.get("start_date"), kwargs.get("end_date")))
            return pd.DataFrame()

    monkeypatch.setattr("quant_investor.market.fundamental_mart.time.sleep", lambda _seconds: None)
    pro = _FakePro()

    tables, manifest = _fetch_tushare_tables(
        ["000001.SZ", "000002.SZ"],
        years=5,
        as_of="20240510",
        workers=4,
        pro=pro,
    )

    assert len(tables["fina_indicator"]) == 1
    assert manifest["requests_failed"] == 1
    assert manifest["errors"][0]["symbol"] == "000002.SZ"
    assert manifest["raw_row_counts"]["fina_indicator"] == 1
    first_start_by_table = {}
    for table, _symbol, start_date, _end_date in pro.calls:
        first_start_by_table.setdefault(table, start_date)
    assert first_start_by_table["daily_basic"] == "20190510"
    assert first_start_by_table["fina_indicator"] == "20170510"
    assert first_start_by_table["income"] == "20170510"
    assert first_start_by_table["forecast"] == "20170510"
    assert manifest["daily_start_date"] == "20190510"
    assert manifest["financial_start_date"] == "20170510"
