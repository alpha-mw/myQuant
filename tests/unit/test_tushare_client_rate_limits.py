from __future__ import annotations

import logging

import pandas as pd
import pytest

from quant_investor.data._tushare_client import TushareClientPool
from quant_investor.data.sources.tushare_cn import TushareDataSource


def _fresh_pool(monkeypatch) -> TushareClientPool:
    monkeypatch.setattr(TushareClientPool, "_instance", None)
    return TushareClientPool()


def test_report_rc_uses_endpoint_local_rate_limit(monkeypatch):
    pool = _fresh_pool(monkeypatch)
    clock = {"now": 1000.0}
    sleeps: list[float] = []
    calls: list[str] = []

    class FakePro:
        def report_rc(self, **kwargs):
            calls.append(kwargs["ts_code"])
            return pd.DataFrame([{"ts_code": kwargs["ts_code"]}])

    monkeypatch.setattr("quant_investor.data._tushare_client.time.monotonic", lambda: clock["now"])

    def _fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr("quant_investor.data._tushare_client.time.sleep", _fake_sleep)
    monkeypatch.setattr(pool, "_get_pro", lambda: FakePro())

    pool.query("report_rc", ts_code="000001.SZ")
    pool.query("report_rc", ts_code="000002.SZ")

    assert calls == ["000001.SZ", "000002.SZ"]
    assert sleeps == [pytest.approx(30.0)]


def test_report_rc_waits_and_retries_on_quota(monkeypatch, caplog):
    pool = _fresh_pool(monkeypatch)
    clock = {"now": 1000.0}
    sleeps: list[float] = []
    attempts = {"count": 0}

    class FakePro:
        def report_rc(self, **kwargs):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("抱歉，您每分钟最多访问该接口2次，请稍后再试")
            return pd.DataFrame([{"ts_code": kwargs["ts_code"], "eps_est": 1.0}])

    monkeypatch.setattr("quant_investor.data._tushare_client.time.monotonic", lambda: clock["now"])

    def _fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr("quant_investor.data._tushare_client.time.sleep", _fake_sleep)
    monkeypatch.setattr(pool, "_get_pro", lambda: FakePro())

    caplog.set_level(logging.INFO, logger="data.TushareClientPool")
    result = pool.query("report_rc", ts_code="000001.SZ", wait_on_quota=True)

    assert not result.empty
    assert attempts["count"] == 2
    assert sleeps == [65.0]
    quota_warnings = [
        record for record in caplog.records
        if "quota hit" in record.getMessage() and record.levelno >= logging.WARNING
    ]
    assert len(quota_warnings) == 1


def test_permission_failure_opens_endpoint_circuit(monkeypatch):
    pool = _fresh_pool(monkeypatch)
    attempts = {"count": 0}

    class FakePro:
        def report_rc(self, **kwargs):
            attempts["count"] += 1
            raise RuntimeError("403 permission denied")

    monkeypatch.setattr(pool, "_get_pro", lambda: FakePro())

    with pytest.raises(RuntimeError):
        pool.query("report_rc", ts_code="000001.SZ")

    with pytest.raises(RuntimeError, match="circuit open for report_rc"):
        pool.query("report_rc", ts_code="000001.SZ")

    assert attempts["count"] == 1


def test_earnings_forecast_snapshot_falls_back_after_report_rc_permission_error(monkeypatch):
    source = TushareDataSource()
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeClient:
        available = True

        def query(self, api_name: str, **kwargs):
            calls.append((api_name, kwargs))
            if api_name == "report_rc":
                raise RuntimeError("403 permission denied")
            return pd.DataFrame(
                [
                    {
                        "net_profit_min": 120.0,
                        "net_profit_max": 180.0,
                        "last_parent_net": 100.0,
                    }
                ]
            )

    source._client = FakeClient()

    snapshot = source.get_earnings_forecast_snapshot("000001.SZ", "2026-03-26")

    assert calls[0][0] == "report_rc"
    assert calls[0][1]["wait_on_quota"] is True
    assert calls[1][0] == "forecast"
    assert snapshot.available is True
    assert snapshot.source == "tushare_forecast"
