"""Research-only DataYes API adapter for source benchmarking.

The adapter deliberately stops at provider-to-canonical normalization.  It does
not publish, activate, or replace the Tushare-backed canonical store.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import requests

OFFICIAL_DATAYES_BASE_URL = "https://api.datayes.com/data/v1/api"
_TOKEN_RE = re.compile(r"^[A-Za-z0-9]{20,128}$", re.ASCII)
_ENDPOINTS = {
    "security_master": ("equity", "getEqu"),
    "daily": ("market", "getMktEqud"),
    "st": ("equity", "getSecST"),
    "suspension": ("master", "getSecHalt"),
    "price_limits": ("market", "getMktLimit"),
    "trade_calendar": ("master", "getTradeCal"),
    "raw_bs_pit": ("fundamental", "getFdmtBS"),
    "raw_is_pit": ("fundamental", "getFdmtIS"),
    "raw_cf_pit": ("fundamental", "getFdmtCF"),
    "quarter_is_pit": ("fundamental", "getFdmtISQPIT"),
    "quarter_cf_pit": ("fundamental", "getFdmtCFQPIT"),
    "ttm_is_pit": ("fundamental", "getFdmtISTTMPIT"),
    "ttm_cf_pit": ("fundamental", "getFdmtCFTTMPIT"),
    "indicator_pit": ("fundamental", "getFdmtMainDataPIT"),
}


class DataYesProviderError(RuntimeError):
    """Sanitized DataYes provider failure that never includes token material."""


@dataclass(frozen=True)
class DataYesResponse:
    dataset: str
    rows: tuple[Mapping[str, Any], ...]
    ret_msg: str


def _token_from_env() -> str:
    token = str(os.environ.get("DATAYES_TOKEN") or "").strip()
    if not _TOKEN_RE.fullmatch(token):
        raise DataYesProviderError("DATAYES_TOKEN_MISSING_OR_INVALID")
    return token


def _date(value: Any) -> str:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else ""


def _ticker_to_ts_code(ticker: Any, exchange: Any = "") -> str:
    value = str(ticker or "").strip().zfill(6)
    market = str(exchange or "").strip().upper()
    suffix = {"XSHG": "SH", "XSHE": "SZ", "XBSE": "BJ"}.get(market)
    if suffix is None:
        if value.startswith(("4", "8", "92")):
            suffix = "BJ"
        elif value.startswith(("5", "6", "9")):
            suffix = "SH"
        else:
            suffix = "SZ"
    return f"{value}.{suffix}"


def _number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def _percent_fraction(value: Any) -> float:
    number = _number(value)
    return number / 100.0 if np.isfinite(number) else np.nan


class DataYesTransport:
    """Small allow-listed HTTP transport for the official DataYes API."""

    def __init__(
        self,
        *,
        token: str | None = None,
        timeout_seconds: float = 25.0,
        session: requests.Session | None = None,
    ) -> None:
        self._token = token or _token_from_env()
        if not _TOKEN_RE.fullmatch(self._token):
            raise DataYesProviderError("DATAYES_TOKEN_MISSING_OR_INVALID")
        self._timeout = float(timeout_seconds)
        self._session = session or requests.Session()
        self.warnings: list[str] = []

    def request(self, dataset: str, params: Mapping[str, Any]) -> DataYesResponse:
        if dataset not in _ENDPOINTS:
            raise DataYesProviderError("DATAYES_DATASET_BLOCKED")
        category, endpoint = _ENDPOINTS[dataset]
        url = f"{OFFICIAL_DATAYES_BASE_URL}/{category}/{endpoint}.json"
        safe_params = {"field": ""}
        safe_params.update({str(k): v for k, v in params.items() if v not in (None, "", [], ())})
        try:
            response = self._session.get(
                url,
                params=safe_params,
                headers={"Authorization": f"Bearer {self._token}"},
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            raise DataYesProviderError("DATAYES_TRANSPORT_ERROR") from exc
        if response.status_code != 200 or len(response.content) > 64 * 1024 * 1024:
            raise DataYesProviderError("DATAYES_HTTP_ERROR")
        try:
            payload = response.json()
        except ValueError as exc:
            raise DataYesProviderError("DATAYES_RESPONSE_INVALID") from exc
        if not isinstance(payload, dict) or not {"retCode", "retMsg"}.issubset(payload):
            raise DataYesProviderError("DATAYES_RESPONSE_INVALID")
        if payload["retCode"] == -1 and payload["retMsg"] == "No Data Returned":
            return DataYesResponse(dataset=dataset, rows=(), ret_msg=payload["retMsg"])
        if set(payload) != {"retCode", "retMsg", "data"}:
            raise DataYesProviderError("DATAYES_RESPONSE_INVALID")
        if payload["retCode"] != 1 or not isinstance(payload["retMsg"], str):
            raise DataYesProviderError(f"DATAYES_API_ERROR:{payload['retCode']}")
        if payload["retMsg"] != "Success":
            warning = f"{dataset}:{payload['retMsg']}"
            if warning not in self.warnings:
                self.warnings.append(warning)
        rows = payload["data"]
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise DataYesProviderError("DATAYES_RESPONSE_INVALID")
        return DataYesResponse(dataset=dataset, rows=tuple(rows), ret_msg=payload["retMsg"])


class DataYesProvider:
    """Map DataYes fields into the existing myQuant canonical vocabulary."""

    def __init__(self, transport: DataYesTransport | None = None) -> None:
        self.transport = transport or DataYesTransport()

    @staticmethod
    def _chunks(values: Iterable[str], size: int = 50) -> Iterable[list[str]]:
        items = list(dict.fromkeys(str(value).split(".", 1)[0] for value in values))
        for start in range(0, len(items), size):
            yield items[start : start + size]

    def _batched(
        self,
        dataset: str,
        symbols: Iterable[str],
        *,
        batch_size: int = 50,
        **params: Any,
    ) -> pd.DataFrame:
        rows: list[Mapping[str, Any]] = []
        for chunk in self._chunks(symbols, size=batch_size):
            result = self.transport.request(dataset, {**params, "ticker": ",".join(chunk)})
            rows.extend(result.rows)
        return pd.DataFrame(rows)

    def security_master(self, symbols: Iterable[str]) -> pd.DataFrame:
        raw = self._batched("security_master", symbols)
        if raw.empty:
            return pd.DataFrame(columns=["ts_code", "name", "list_date", "delist_date"])
        return pd.DataFrame(
            {
                "ts_code": [
                    _ticker_to_ts_code(row.get("ticker"), row.get("exchangeCD"))
                    for row in raw.to_dict("records")
                ],
                "name": raw.get("secShortName", pd.Series(index=raw.index, dtype=object)),
                "list_date": raw.get("listDate", pd.Series(index=raw.index, dtype=object)).map(
                    _date
                ),
                "delist_date": raw.get("delistDate", pd.Series(index=raw.index, dtype=object)).map(
                    _date
                ),
            }
        )

    def daily(self, symbols: Iterable[str], *, start_date: str, end_date: str) -> pd.DataFrame:
        raw = self._batched("daily", symbols, beginDate=_date(start_date), endDate=_date(end_date))
        if raw.empty:
            return pd.DataFrame()
        records = raw.to_dict("records")
        frame = pd.DataFrame(
            {
                "ts_code": [
                    _ticker_to_ts_code(row.get("ticker"), row.get("exchangeCD")) for row in records
                ],
                "trade_date": [_date(row.get("tradeDate")) for row in records],
                "open": [_number(row.get("openPrice")) for row in records],
                "high": [_number(row.get("highestPrice")) for row in records],
                "low": [_number(row.get("lowestPrice")) for row in records],
                "close": [_number(row.get("closePrice")) for row in records],
                "pre_close": [_number(row.get("actPreClosePrice")) for row in records],
                "pct_chg": [_number(row.get("chgPct")) * 100.0 for row in records],
                "vol": [_number(row.get("turnoverVol")) / 100.0 for row in records],
                "amount": [_number(row.get("turnoverValue")) / 1000.0 for row in records],
                "adj_factor": [_number(row.get("accumAdjFactor")) for row in records],
                "turnover_rate": [_number(row.get("turnoverRate")) * 100.0 for row in records],
                "pe": [_number(row.get("PE")) for row in records],
                "pb": [_number(row.get("PB")) for row in records],
                "total_mv": [_number(row.get("marketValue")) / 10000.0 for row in records],
                "circ_mv": [_number(row.get("negMarketValue")) / 10000.0 for row in records],
                "is_open": [int(row.get("isOpen") or 0) for row in records],
            }
        )
        frame["change"] = frame["close"] - frame["pre_close"]
        frame["adj_close"] = frame["close"] * frame["adj_factor"]
        frame["adj_open"] = frame["open"] * frame["adj_factor"]
        frame["adj_high"] = frame["high"] * frame["adj_factor"]
        frame["adj_low"] = frame["low"] * frame["adj_factor"]
        return frame.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)

    def adjustment_factor(
        self, symbols: Iterable[str], *, start_date: str, end_date: str
    ) -> pd.DataFrame:
        return self.daily(symbols, start_date=start_date, end_date=end_date)[
            ["ts_code", "trade_date", "adj_factor"]
        ]

    def st(self, symbols: Iterable[str], *, start_date: str, end_date: str) -> pd.DataFrame:
        return self._batched("st", symbols, beginDate=_date(start_date), endDate=_date(end_date))

    def suspension(self, symbols: Iterable[str], *, start_date: str, end_date: str) -> pd.DataFrame:
        return self._batched(
            "suspension", symbols, beginDate=_date(start_date), endDate=_date(end_date)
        )

    def price_limits(self, symbols: Iterable[str], *, trade_date: str) -> pd.DataFrame:
        raw = self._batched("price_limits", symbols, tradeDate=_date(trade_date))
        if raw.empty:
            return raw
        raw = raw.copy()
        raw["ts_code"] = [
            _ticker_to_ts_code(row.get("ticker"), row.get("exchangeCD"))
            for row in raw.to_dict("records")
        ]
        raw["trade_date"] = raw["tradeDate"].map(_date)
        return raw.rename(columns={"limitUpPrice": "up_limit", "limitDownPrice": "down_limit"})

    def trade_calendar(self, *, start_date: str, end_date: str) -> pd.DataFrame:
        raw = pd.DataFrame(
            self.transport.request(
                "trade_calendar",
                {
                    "exchangeCD": "XSHG,XSHE",
                    "beginDate": _date(start_date),
                    "endDate": _date(end_date),
                },
            ).rows
        )
        if raw.empty:
            return raw
        return raw.rename(columns={"calendarDate": "cal_date", "isOpen": "is_open"})

    def fundamental(
        self,
        dataset: str,
        symbols: Iterable[str],
        *,
        start_date: str,
        end_date: str,
        fields: str = "",
        batch_size: int = 10,
    ) -> pd.DataFrame:
        if dataset not in {
            "raw_bs_pit",
            "raw_is_pit",
            "raw_cf_pit",
            "quarter_is_pit",
            "quarter_cf_pit",
            "ttm_is_pit",
            "ttm_cf_pit",
            "indicator_pit",
        }:
            raise DataYesProviderError("DATAYES_FUNDAMENTAL_DATASET_BLOCKED")
        raw = self._batched(
            dataset,
            symbols,
            batch_size=batch_size,
            field=fields,
            beginDate=_date(start_date),
            endDate=_date(end_date),
        )
        if raw.empty:
            return raw
        raw = raw.copy()
        records = raw.to_dict("records")
        raw["ts_code"] = [
            _ticker_to_ts_code(row.get("ticker"), row.get("exchangeCD")) for row in records
        ]
        if "endDate" in raw:
            raw["end_date"] = raw["endDate"].map(_date)
        if "publishDate" in raw:
            raw["availability_date"] = raw["publishDate"].map(_date)
        return raw

    def indicator_pit(
        self, symbols: Iterable[str], *, start_date: str, end_date: str
    ) -> pd.DataFrame:
        raw = self.fundamental(
            "indicator_pit",
            symbols,
            start_date=start_date,
            end_date=end_date,
            batch_size=20,
            fields=(
                "ticker,exchangeCD,endDate,publishDate,roeCut,roaEbit," "assetLiabRatio,niAttrPYoy"
            ),
        )
        if raw.empty:
            return raw
        return pd.DataFrame(
            {
                "ts_code": raw["ts_code"],
                "end_date": raw["end_date"],
                "availability_date": raw["availability_date"],
                "fin_roe": raw.get("roeCut", raw.get("roe")).map(_percent_fraction),
                "fin_roa": raw.get("roaEbit", raw.get("roa")).map(_percent_fraction),
                "fin_debt_to_assets": raw.get("assetLiabRatio").map(_percent_fraction),
                "fin_net_profit_yoy": raw.get("niAttrPYoy", raw.get("niYoy")).map(
                    _percent_fraction
                ),
            }
        )


__all__ = [
    "DataYesProvider",
    "DataYesProviderError",
    "DataYesResponse",
    "DataYesTransport",
    "OFFICIAL_DATAYES_BASE_URL",
]
