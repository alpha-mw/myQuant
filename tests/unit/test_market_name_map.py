"""Market symbol-name map cache tests."""

from __future__ import annotations

import json

from quant_investor.market import name_map


def test_load_company_name_map_reads_local_cn_cache_without_provider(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    cache_path = tmp_path / "data" / "cn_universe" / "stock_names.json"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(
        json.dumps({"000001.sz": "平安银行", "": "ignored"}),
        encoding="utf-8",
    )

    def _raise_provider(*args, **kwargs):
        raise AssertionError("provider should not be used")

    monkeypatch.setattr(name_map, "create_tushare_pro", _raise_provider)
    name_map.clear_stock_name_cache()

    names = name_map.load_company_name_map("CN", allow_provider=False)

    assert names == {"000001.SZ": "平安银行"}


def test_load_us_stock_names_builds_cache_from_local_market_cap(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    source_path = tmp_path / "data" / "us_universe" / "us_market_caps.json"
    cache_path = tmp_path / "data" / "us_universe" / "stock_names.json"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        json.dumps({"symbols": {"bce": {"name": "BCE Inc."}}}),
        encoding="utf-8",
    )
    name_map.clear_stock_name_cache("US")

    names = name_map.load_stock_names("US", refresh=True)

    assert names == {"BCE": "BCE Inc."}
    assert json.loads(cache_path.read_text(encoding="utf-8")) == {
        "BCE": "BCE Inc.",
    }


def test_get_stock_name_uses_memory_cache_before_disk(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    cache_path = tmp_path / "data" / "cn_universe" / "stock_names.json"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(
        json.dumps({"000001.SZ": "平安银行"}),
        encoding="utf-8",
    )
    name_map.clear_stock_name_cache("CN")

    assert name_map.get_stock_name("000001.SZ", market="CN") == "平安银行"

    cache_path.write_text(
        json.dumps({"000001.SZ": "SHOULD_NOT_RELOAD"}),
        encoding="utf-8",
    )

    assert name_map.get_stock_name("000001.SZ", market="CN") == "平安银行"
