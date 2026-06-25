from __future__ import annotations

import pandas as pd

import quant_investor.market.dag_executor as dag_executor


def test_load_company_profile_map_uses_canonical_parquet_stock_basic(
    tmp_path,
    monkeypatch,
):
    data_root = tmp_path / "data"
    table_dir = data_root / "parquet" / "cn" / "dag_core_raw" / "table=stock_basic"
    table_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "name": "平安银行",
                "industry": "银行",
            },
            {
                "ts_code": "002920.SZ",
                "name": "德赛西威",
                "industry": "汽车配件",
            },
        ]
    ).to_parquet(table_dir / "part.parquet", index=False)
    monkeypatch.setattr(dag_executor.config, "DATA_DIR", str(data_root))
    monkeypatch.setattr(dag_executor.config, "DB_PATH", str(tmp_path / "missing.db"))

    profiles = dag_executor._load_company_profile_map("CN")

    assert profiles["000001.SZ"]["name"] == "平安银行"
    assert profiles["000001.SZ"]["industry"] == "银行"
    assert profiles["000001.SZ"]["sector"] == "银行"
    assert profiles["002920.SZ"]["industry"] == "汽车配件"
    assert profiles["002920.SZ"]["profile_source"] == "canonical_parquet_stock_basic"
