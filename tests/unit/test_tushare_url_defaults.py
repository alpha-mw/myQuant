from __future__ import annotations

import importlib
import sys
import types


def _reload_module(name: str):
    sys.modules.pop(name, None)
    return importlib.import_module(name)


def _install_fake_tushare(monkeypatch) -> None:
    module = types.ModuleType("tushare")
    module.pro_api = lambda token: object()
    monkeypatch.setitem(sys.modules, "tushare", module)


def test_tushare_default_url_is_new_proxy(monkeypatch):
    monkeypatch.delenv("TUSHARE_URL", raising=False)
    monkeypatch.delenv("TUSHARE_RATE_LIMIT_PER_MIN", raising=False)
    _install_fake_tushare(monkeypatch)

    config_module = _reload_module("quant_investor.config")
    registry_module = _reload_module("quant_investor.data._registry")
    pool_module = _reload_module("quant_investor.data._tushare_client")
    fetch_module = _reload_module("quant_investor.fetch_cn_index_components")
    download_module = _reload_module("quant_investor.market.download_cn")
    stock_database_module = _reload_module("quant_investor.stock_database")

    expected = "http://139.196.25.182"

    assert config_module.MAINLINE_ENV_DEFAULTS["TUSHARE_URL"] == expected
    assert config_module.MAINLINE_ENV_DEFAULTS["TUSHARE_RATE_LIMIT_PER_MIN"] == "500"
    assert config_module.Config.TUSHARE_URL == expected
    assert config_module.Config.TUSHARE_RATE_LIMIT_PER_MIN == 500
    assert registry_module.TUSHARE_CATALOG["daily"].rate_limit_per_min == 500
    assert fetch_module.TUSHARE_URL == expected
    assert download_module.TUSHARE_URL == expected
    assert download_module.CNFullMarketDownloader.REQUESTS_PER_MINUTE_BUDGET == 500
    assert stock_database_module.DEFAULT_TUSHARE_URL == expected
    pool_module.TushareClientPool._instance = None
    pool = pool_module.TushareClientPool()
    pool._ensure_config()
    assert pool._url == expected


def test_tushare_url_still_allows_env_override(monkeypatch):
    override = "http://127.0.0.1:9000"
    override_rate = "420"
    monkeypatch.setenv("TUSHARE_URL", override)
    monkeypatch.setenv("TUSHARE_RATE_LIMIT_PER_MIN", override_rate)
    _install_fake_tushare(monkeypatch)

    config_module = _reload_module("quant_investor.config")
    registry_module = _reload_module("quant_investor.data._registry")
    pool_module = _reload_module("quant_investor.data._tushare_client")
    fetch_module = _reload_module("quant_investor.fetch_cn_index_components")
    download_module = _reload_module("quant_investor.market.download_cn")
    stock_database_module = _reload_module("quant_investor.stock_database")

    assert config_module.Config.TUSHARE_URL == override
    assert config_module.Config.TUSHARE_RATE_LIMIT_PER_MIN == 420
    assert registry_module.TUSHARE_CATALOG["daily"].rate_limit_per_min == 420
    assert fetch_module.TUSHARE_URL == override
    assert download_module.TUSHARE_URL == override
    assert download_module.CNFullMarketDownloader.REQUESTS_PER_MINUTE_BUDGET == 420
    assert stock_database_module.DEFAULT_TUSHARE_URL == override
    pool_module.TushareClientPool._instance = None
    pool = pool_module.TushareClientPool()
    pool._ensure_config()
    assert pool._url == override
