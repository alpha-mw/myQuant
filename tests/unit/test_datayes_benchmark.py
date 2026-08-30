from __future__ import annotations

import pandas as pd
import pytest

from quant_investor.market.datayes_provider import (
    DataYesProvider,
    DataYesProviderError,
    DataYesTransport,
)
from quant_investor.research.data_source_benchmark import (
    compare_candidates,
    compare_factors,
    compare_frames,
    compare_rankic,
    procurement_decision,
    rank_combined_signals,
)
from scripts.run_datayes_benchmark import _cached_frame, _tushare_indicator_from_raw


class _Response:
    status_code = 200
    content = b"{}"

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _Session:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _Response(self.payload)


def test_transport_uses_bearer_and_allow_list_without_token_in_params():
    session = _Session({"retCode": 1, "retMsg": "Success", "data": []})
    transport = DataYesTransport(token="A" * 64, session=session)
    transport.request("daily", {"ticker": "000001"})
    url, kwargs = session.calls[0]
    assert url == "https://api.datayes.com/data/v1/api/market/getMktEqud.json"
    assert kwargs["headers"] == {"Authorization": "Bearer " + "A" * 64}
    assert "token" not in kwargs["params"]
    with pytest.raises(DataYesProviderError, match="DATAYES_DATASET_BLOCKED"):
        transport.request("arbitrary", {})


def test_transport_accepts_official_no_data_envelope_without_data_key():
    session = _Session({"retCode": -1, "retMsg": "No Data Returned"})
    response = DataYesTransport(token="A" * 64, session=session).request("st", {"ticker": "000001"})
    assert response.rows == ()


def test_daily_maps_units_to_existing_canonical_schema():
    payload = {
        "retCode": 1,
        "retMsg": "Success",
        "data": [
            {
                "ticker": "000001",
                "exchangeCD": "XSHE",
                "tradeDate": "2026-08-27",
                "openPrice": 10,
                "highestPrice": 11,
                "lowestPrice": 9,
                "closePrice": 10.5,
                "actPreClosePrice": 10,
                "chgPct": 0.05,
                "turnoverVol": 10000,
                "turnoverValue": 200000,
                "accumAdjFactor": 1.2,
                "turnoverRate": 0.01,
                "PE": 5,
                "PB": 1,
                "marketValue": 100000000,
                "negMarketValue": 80000000,
                "isOpen": 1,
            }
        ],
    }
    provider = DataYesProvider(DataYesTransport(token="A" * 64, session=_Session(payload)))
    row = provider.daily(["000001.SZ"], start_date="20260827", end_date="20260827").iloc[0]
    assert row.ts_code == "000001.SZ"
    assert row.trade_date == "20260827"
    assert row.vol == 100
    assert row.amount == 200
    assert row.turnover_rate == 1
    assert row.total_mv == 10000
    assert row.adj_close == 12.6


def test_indicator_percentages_map_to_canonical_fractions():
    payload = {
        "retCode": 1,
        "retMsg": "Success",
        "data": [
            {
                "ticker": "000001",
                "exchangeCD": "XSHE",
                "endDate": "20260331",
                "publishDate": "20260425",
                "roeCut": 8.1,
                "roaEbit": 1.2,
                "assetLiabRatio": 90.5,
                "niAttrPYoy": -4.2,
            }
        ],
    }
    provider = DataYesProvider(DataYesTransport(token="A" * 64, session=_Session(payload)))
    row = provider.indicator_pit(["000001.SZ"], start_date="20260101", end_date="20260827").iloc[0]
    assert row.fin_roe == pytest.approx(0.081)
    assert row.fin_roa == pytest.approx(0.012)
    assert row.fin_debt_to_assets == pytest.approx(0.905)
    assert row.fin_net_profit_yoy == pytest.approx(-0.042)


def test_indicator_requests_use_projected_fields_and_twenty_symbol_batches():
    session = _Session({"retCode": 1, "retMsg": "Success", "data": []})
    provider = DataYesProvider(DataYesTransport(token="A" * 64, session=session))
    provider.indicator_pit(
        [f"{index:06d}.SZ" for index in range(21)],
        start_date="20250101",
        end_date="20260827",
    )
    assert len(session.calls) == 2
    assert session.calls[0][1]["params"]["field"] == (
        "ticker,exchangeCD,endDate,publishDate,roeCut,roaEbit," "assetLiabRatio,niAttrPYoy"
    )


def test_private_cache_reuses_completed_frame(tmp_path):
    calls = []

    def fetch():
        calls.append(True)
        return pd.DataFrame({"ts_code": ["000001.SZ"], "close": [10.0]})

    first, first_hit = _cached_frame(tmp_path, request_sha256="a" * 64, name="market", fetch=fetch)
    second, second_hit = _cached_frame(
        tmp_path, request_sha256="a" * 64, name="market", fetch=fetch
    )
    assert not first_hit
    assert second_hit
    assert calls == [True]
    pd.testing.assert_frame_equal(first, second)


def test_tushare_raw_percentages_are_unconditionally_normalized(tmp_path):
    path = tmp_path / "fina_indicator.parquet"
    pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "ann_date": [20260425.0],
            "end_date": [20260331],
            "roe_dt": [1.5],
            "roe": [1.6],
            "roa": [1.9997],
            "debt_to_assets": [1.8303],
            "netprofit_yoy": [-1.25],
        }
    ).to_parquet(path, index=False)
    row = _tushare_indicator_from_raw(path, symbols=("000001.SZ",), end_date="20260827").iloc[0]
    assert row.fin_roe == pytest.approx(0.015)
    assert row.fin_roa == pytest.approx(0.019997)
    assert row.fin_debt_to_assets == pytest.approx(0.018303)
    assert row.fin_net_profit_yoy == pytest.approx(-0.0125)


def test_combined_rank_candidates_and_rankic_comparison():
    signals = {
        "low": pd.Series({"A": 1.0, "B": 2.0, "C": 3.0}),
        "w80": pd.Series({"A": 3.0, "B": 2.0, "C": 1.0}),
    }
    rows = rank_combined_signals(signals, weights={"low": 0.5, "w80": 0.5})
    assert [row["symbol"] for row in rows] == ["A", "B", "C"]
    candidates = compare_candidates(rows, list(reversed(rows)), top_n=2)
    assert candidates["overlap"] == 0.5
    rankic = compare_rankic(
        {"f": {"20260101": 0.1, "20260201": 0.2}},
        {"f": {"20260101": 0.1, "20260201": 0.21}},
    )
    assert rankic["f"]["anchor_count"] == 2
    assert rankic["f"]["systematic_improvement"] is False


def test_metrics_and_procurement_fail_closed_without_twenty_factors_or_rankic():
    left = pd.DataFrame({"ts_code": ["A"], "trade_date": ["1"], "close": [1.0]})
    right = pd.DataFrame({"ts_code": ["A"], "trade_date": ["1"], "close": [1.0]})
    market = compare_frames(left, right, keys=("ts_code", "trade_date"), fields=("close",))
    fundamental = {"fields": {"roe": {"status": "COMPARED", "spearman": 1.0}}}
    factors = compare_factors({"f": pd.Series({"A": 1.0})}, {"f": pd.Series({"A": 1.0})})
    decision = procurement_decision(
        market=market, fundamental=fundamental, factors=factors, rankic={}
    )
    assert market["fields"]["close"]["largest_differences"] == [
        {
            "ts_code": "A",
            "trade_date": "1",
            "tushare": 1.0,
            "datayes": 1.0,
            "absolute_difference": 0.0,
        }
    ]
    assert decision["decision"] == "INSUFFICIENT_EVIDENCE"
    assert decision["purchase_recommendation"] == "DEFER_PURCHASE"
