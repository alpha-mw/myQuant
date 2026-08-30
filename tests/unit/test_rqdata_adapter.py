from __future__ import annotations

import pandas as pd
import pytest

from quant_investor.data._registry import get_endpoint_spec, get_provider_endpoint_spec
from quant_investor.market.rqdata_adapter import (
    RQDataNormalizationError,
    normalize_rqdata_daily_bars,
    normalize_rqdata_symbol,
)
from quant_investor.market.market_data_store import MarketDataStore


def _official_shape_frame() -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [
            ("000001.XSHE", pd.Timestamp("2019-04-01")),
            ("600000.XSHG", pd.Timestamp("2019-04-01")),
        ],
        names=["order_book_id", "date"],
    )
    return pd.DataFrame(
        {
            "open": [12.83, 11.70],
            "high": [13.55, 11.90],
            "low": [12.83, 11.60],
            "close": [13.18, 11.80],
            "volume": [195140119.0, 1000.0],
            "total_turnover": [2588268668.0, 11800.0],
            "prev_close": [12.82, 11.60],
            "limit_up": [14.10, 12.76],
            "limit_down": [11.54, 10.44],
            "num_trades": [79511.0, 15.0],
        },
        index=index,
    )


def test_registry_adds_rqdata_as_non_authorizing_candidate() -> None:
    rqdata = get_provider_endpoint_spec("rqdata", "get_price")
    tushare = get_provider_endpoint_spec("tushare", "daily")

    assert rqdata.role == "primary_candidate"
    assert rqdata.activation_authorized is False
    assert tushare.role == "production_primary"
    assert tushare.activation_authorized is False
    assert get_endpoint_spec("daily").source_priority == "tushare_primary"
    with pytest.raises(KeyError, match="unregistered tushare endpoint"):
        get_provider_endpoint_spec("tushare", "typo")


@pytest.mark.parametrize(
    ("provider_symbol", "canonical_symbol"),
    [
        ("000001.XSHE", "000001.SZ"),
        ("600000.XSHG", "600000.SH"),
        ("920001.XBSE", "920001.BJ"),
    ],
)
def test_rqdata_symbol_mapping_is_exact(provider_symbol: str, canonical_symbol: str) -> None:
    assert normalize_rqdata_symbol(provider_symbol) == canonical_symbol


def test_daily_adapter_matches_existing_bar_schema_without_unit_scaling() -> None:
    normalized = normalize_rqdata_daily_bars(_official_shape_frame(), adjustment_type="none")

    assert normalized["ts_code"].tolist() == ["000001.SZ", "600000.SH"]
    assert normalized["trade_date"].tolist() == ["20190401", "20190401"]
    assert normalized["vol"].tolist() == [195140119.0, 1000.0]
    assert normalized["amount"].tolist() == [2588268668.0, 11800.0]
    assert normalized["provider_symbol"].tolist() == ["000001.XSHE", "600000.XSHG"]
    assert normalized["adjustment_type"].unique().tolist() == ["none"]
    assert normalized["bar_status"].unique().tolist() == ["TRADING"]

    store_ready = MarketDataStore(data_root="unused")._normalize_bars_frame(normalized)
    assert store_ready["vol"].tolist() == [195140119.0, 1000.0]


@pytest.mark.parametrize("adjustment_type", ["pre", "post", ""])
def test_daily_adapter_rejects_adjusted_prices(adjustment_type: str) -> None:
    with pytest.raises(RQDataNormalizationError, match="adjust_type='none'"):
        normalize_rqdata_daily_bars(_official_shape_frame(), adjustment_type=adjustment_type)


def test_daily_adapter_fails_whole_response_on_unknown_symbol_or_duplicate() -> None:
    unknown = _official_shape_frame().reset_index()
    unknown.loc[0, "order_book_id"] = "000001.UNKNOWN"
    with pytest.raises(RQDataNormalizationError, match="unsupported RQData symbol"):
        normalize_rqdata_daily_bars(unknown, adjustment_type="none")

    duplicate = pd.concat([_official_shape_frame(), _official_shape_frame().iloc[[0]]])
    with pytest.raises(RQDataNormalizationError, match="duplicate"):
        normalize_rqdata_daily_bars(duplicate, adjustment_type="none")


def test_daily_adapter_keeps_complete_no_trade_row_explicit() -> None:
    frame = _official_shape_frame().iloc[[0]].copy()
    frame.loc[:, ["open", "high", "low", "close"]] = float("nan")
    frame.loc[:, ["volume", "total_turnover"]] = 0.0

    normalized = normalize_rqdata_daily_bars(frame, adjustment_type="none")

    assert normalized.loc[0, "bar_status"] == "SUSPENDED_OR_NO_TRADE"


def test_daily_adapter_rejects_partial_ohlc_and_negative_volume() -> None:
    partial = _official_shape_frame().iloc[[0]].copy()
    partial.loc[:, "open"] = float("nan")
    with pytest.raises(RQDataNormalizationError, match="partial OHLC"):
        normalize_rqdata_daily_bars(partial, adjustment_type="none")

    negative = _official_shape_frame().iloc[[0]].copy()
    negative.loc[:, "volume"] = -1
    with pytest.raises(RQDataNormalizationError, match="negative"):
        normalize_rqdata_daily_bars(negative, adjustment_type="none")


def test_daily_adapter_rejects_empty_response() -> None:
    with pytest.raises(RQDataNormalizationError, match="empty"):
        normalize_rqdata_daily_bars(_official_shape_frame().iloc[0:0], adjustment_type="none")
