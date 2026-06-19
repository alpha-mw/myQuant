from __future__ import annotations

import pandas as pd

from quant_investor.factors.pit_fundamentals import (
    PIT_COLUMNS,
    append_tushare_financial_pit_series,
    build_fin_ocf_to_profit_matrix,
    load_fundamental_pit_series,
)


def test_fin_ocf_to_profit_visible_only_after_availability_and_zero_profit_nan(tmp_path):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "report_period": "20231231",
                "availability_date": "2024-04-30",
                "metric_name": "operating_cashflow",
                "value": 100.0,
                "source": "fixture",
                "fetched_at": "2024-05-10T00:00:00Z",
                "raw_table": "cashflow",
                "raw_field": "n_cashflow_act",
            },
            {
                "ts_code": "000001.SZ",
                "report_period": "20231231",
                "availability_date": "2024-04-30",
                "metric_name": "net_income",
                "value": 50.0,
                "source": "fixture",
                "fetched_at": "2024-05-10T00:00:00Z",
                "raw_table": "income",
                "raw_field": "n_income",
            },
            {
                "ts_code": "000002.SZ",
                "report_period": "20231231",
                "availability_date": "2024-04-30",
                "metric_name": "operating_cashflow",
                "value": 100.0,
                "source": "fixture",
                "fetched_at": "2024-05-10T00:00:00Z",
                "raw_table": "cashflow",
                "raw_field": "n_cashflow_act",
            },
            {
                "ts_code": "000002.SZ",
                "report_period": "20231231",
                "availability_date": "2024-04-30",
                "metric_name": "net_income",
                "value": 0.0,
                "source": "fixture",
                "fetched_at": "2024-05-10T00:00:00Z",
                "raw_table": "income",
                "raw_field": "n_income",
            },
        ],
        columns=PIT_COLUMNS,
    ).to_parquet(metadata_dir / "fundamental_pit_series.parquet", index=False)

    dates = pd.to_datetime(["2024-04-29", "2024-04-30", "2024-05-02"])
    matrix, diagnostics = build_fin_ocf_to_profit_matrix(
        dates,
        ["000001.SZ", "000002.SZ"],
        metadata_dir=metadata_dir,
        mart_root=tmp_path / "missing_mart",
    )

    assert pd.isna(matrix.loc[pd.Timestamp("2024-04-29"), "000001.SZ"])
    assert matrix.loc[pd.Timestamp("2024-04-30"), "000001.SZ"] == 2.0
    assert matrix.loc[pd.Timestamp("2024-05-02"), "000001.SZ"] == 2.0
    assert pd.isna(matrix.loc[pd.Timestamp("2024-04-30"), "000002.SZ"])
    assert diagnostics.symbols_with_ocf_profit == ["000001.SZ"]


def test_repeated_direct_revision_deduplicates_by_availability(tmp_path):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "report_period": "20231231",
                "availability_date": "2024-04-30",
                "metric_name": "fin_ocf_to_profit",
                "value": 1.0,
                "source": "a",
                "fetched_at": "2024-05-01T00:00:00Z",
                "raw_table": "derived",
                "raw_field": "fin_ocf_to_profit",
            },
            {
                "ts_code": "000001.SZ",
                "report_period": "20231231",
                "availability_date": "2024-04-30",
                "metric_name": "fin_ocf_to_profit",
                "value": 2.0,
                "source": "b",
                "fetched_at": "2024-05-02T00:00:00Z",
                "raw_table": "derived",
                "raw_field": "fin_ocf_to_profit",
            },
        ],
        columns=PIT_COLUMNS,
    ).to_parquet(metadata_dir / "fundamental_pit_series.parquet", index=False)

    matrix, _diagnostics = build_fin_ocf_to_profit_matrix(
        pd.to_datetime(["2024-04-30"]),
        ["000001.SZ"],
        metadata_dir=metadata_dir,
        mart_root=tmp_path / "missing_mart",
    )

    assert matrix.loc[pd.Timestamp("2024-04-30"), "000001.SZ"] == 2.0


def test_fin_ocf_to_profit_prefers_canonical_daily_mart(tmp_path):
    mart_root = tmp_path / "parquet" / "cn"
    daily_root = mart_root / "fundamental_daily"
    daily_root.mkdir(parents=True)
    pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "trade_date": "2024-04-30", "fin_ocf_to_profit": 3.0},
            {"ts_code": "000001.SZ", "trade_date": "2024-05-02", "fin_ocf_to_profit": 4.0},
        ]
    ).to_parquet(daily_root / "part.parquet", index=False)
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "report_period": "20231231",
                "availability_date": "2024-04-30",
                "metric_name": "fin_ocf_to_profit",
                "value": 99.0,
                "source": "legacy",
                "fetched_at": "2024-05-02T00:00:00Z",
                "raw_table": "legacy",
                "raw_field": "fin_ocf_to_profit",
            }
        ],
        columns=PIT_COLUMNS,
    ).to_parquet(metadata_dir / "fundamental_pit_series.parquet", index=False)

    matrix, diagnostics = build_fin_ocf_to_profit_matrix(
        pd.to_datetime(["2024-04-30", "2024-05-02"]),
        ["000001.SZ"],
        metadata_dir=metadata_dir,
        mart_root=mart_root,
        allow_legacy_fallback=False,
    )

    assert matrix.loc[pd.Timestamp("2024-04-30"), "000001.SZ"] == 3.0
    assert matrix.loc[pd.Timestamp("2024-05-02"), "000001.SZ"] == 4.0
    assert diagnostics.pit_rows == 2
    assert diagnostics.ratio_rows == 2
    assert diagnostics.symbols_with_ocf_profit == ["000001.SZ"]


class _FakeTusharePro:
    def income(self, **_kwargs):
        return pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "ann_date": "20240428",
                    "f_ann_date": "20240430",
                    "end_date": "20231231",
                    "n_income": 50.0,
                }
            ]
        )

    def cashflow(self, **_kwargs):
        return pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "ann_date": "20240428",
                    "f_ann_date": "20240430",
                    "end_date": "20231231",
                    "n_cashflow_act": 100.0,
                }
            ]
        )


def test_tushare_backfill_writes_pit_rows_without_token(tmp_path):
    metadata_dir = tmp_path / "metadata"
    manifest = append_tushare_financial_pit_series(
        ["000001.SZ"],
        start_date="20230101",
        end_date="20241231",
        metadata_dir=metadata_dir,
        pro=_FakeTusharePro(),
    )

    assert manifest["status"] == "ok"
    assert manifest["token_persisted"] is False
    assert "TUSHARE_TOKEN" not in str(manifest)
    frame = load_fundamental_pit_series(metadata_dir=metadata_dir, mart_root=tmp_path / "missing_mart")
    assert list(frame.columns) == PIT_COLUMNS
    assert set(frame["metric_name"]) == {"net_income", "operating_cashflow"}
    assert frame["availability_date"].dt.strftime("%Y-%m-%d").unique().tolist() == ["2024-04-30"]
