"""CN Tushare source implementation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from quant_investor.branch_contracts import ForecastSnapshot
from quant_investor.data._tushare_client import TushareClientPool
from quant_investor.data.models import FundamentalData
from quant_investor.data.sources.base import DataSourceBase, _filter_ohlcv_by_date, _normalize_ohlcv_frame
from quant_investor.forecast_snapshot_store import ForecastSnapshotStore


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    return number if pd.notna(number) else None


def _normalize_symbol(symbol: str) -> str:
    text = str(symbol or "").strip().upper()
    if "." in text:
        return text
    if text.startswith(("6", "9")):
        return f"{text}.SH"
    if text:
        return f"{text}.SZ"
    return text


class TushareDataSource(DataSourceBase):
    source_name = "tushare_primary"

    def __init__(self, allow_live: bool = True) -> None:
        self._client = TushareClientPool()
        self.allow_live = bool(allow_live)
        self._logger = logging.getLogger("data.sources.TushareCN")
        self.last_ohlcv_source = "unknown"
        self.last_fundamental_source = "unknown"
        self.last_daily_basic_status = "unknown"
        self.last_daily_basic_source = "unknown"
        self.last_daily_basic_reason = ""
        self._tushare = self
        self._forecast_store = ForecastSnapshotStore(
            Path(__file__).resolve().parents[3]
            / "data"
            / "cn_market_full"
            / "_snapshots"
            / "forecast"
        )

    def get_ohlcv(self, symbol: str, start_date: str = "", end_date: str = "", freq: str = "1d") -> pd.DataFrame:
        if not self.allow_live:
            self.last_ohlcv_source = "tushare_live_disabled"
            return pd.DataFrame()
        try:
            frame = self._client.query(
                "daily",
                ts_code=_normalize_symbol(symbol),
                start_date=start_date,
                end_date=end_date,
            )
        except Exception as exc:
            self.last_ohlcv_source = "tushare_daily_error"
            self._logger.warning("daily query failed for %s: %s", symbol, exc)
            return pd.DataFrame()
        normalized = _normalize_ohlcv_frame(frame)
        self.last_ohlcv_source = "tushare_daily" if not normalized.empty else "tushare_daily_empty"
        return _filter_ohlcv_by_date(normalized, start_date, end_date)

    def get_fundamental(self, symbol: str) -> FundamentalData:
        ts_code = _normalize_symbol(symbol)
        if not self.allow_live:
            self.last_fundamental_source = "tushare_live_disabled"
            return FundamentalData(symbol=ts_code, source=self.last_fundamental_source)
        try:
            frame = self._client.query("fina_indicator", ts_code=ts_code)
        except Exception as exc:
            self.last_fundamental_source = "tushare_fina_indicator_error"
            self._logger.warning("fina_indicator query failed for %s: %s", symbol, exc)
            return FundamentalData(symbol=ts_code, source=self.last_fundamental_source)
        if frame is None or frame.empty:
            self.last_fundamental_source = "tushare_fina_indicator_empty"
            return FundamentalData(symbol=ts_code, source=self.last_fundamental_source)
        if "ann_date" in frame.columns:
            frame = frame.sort_values("ann_date", ascending=False)
        row = frame.iloc[0]
        self.last_fundamental_source = "tushare_fina_indicator"
        return FundamentalData(
            symbol=ts_code,
            report_date=str(row.get("ann_date", "") or row.get("end_date", "")),
            roe=_safe_float(row.get("roe_dt")) or _safe_float(row.get("roe")),
            roa=_safe_float(row.get("roa")),
            gross_margin=_safe_float(row.get("grossprofit_margin")),
            net_margin=_safe_float(row.get("netprofit_margin")),
            revenue_growth=_safe_float(row.get("tr_yoy")),
            profit_growth=_safe_float(row.get("netprofit_yoy")),
            debt_ratio=_safe_float(row.get("debt_to_assets")),
            current_ratio=_safe_float(row.get("current_ratio")),
            cash_flow=_safe_float(row.get("ocf_to_profit")),
            source=self.last_fundamental_source,
        )

    @staticmethod
    def _classify_error(exc: Exception) -> str:
        text = str(exc).lower()
        if "circuit open" in text:
            return "circuit_open"
        if "timeout" in text or "timed out" in text:
            return "timeout"
        if any(token in text for token in ("permission", "403", "invalid token", "无效的 token")):
            return "permission_error"
        return "provider_error"

    def get_daily_basic(self, symbol: str, trade_date: str | None = None) -> dict[str, Any]:
        self.last_daily_basic_source = "tushare_daily_basic"
        if not self.allow_live:
            self.last_daily_basic_status = "provider_unavailable"
            self.last_daily_basic_reason = "tushare_live_disabled"
            return {}
        try:
            frame = self._client.query(
                "daily_basic",
                ts_code=_normalize_symbol(symbol),
                trade_date=trade_date or None,
                fields="ts_code,trade_date,pe,pb,ps,dv_ratio,total_mv,circ_mv",
            )
        except Exception as exc:
            self.last_daily_basic_status = self._classify_error(exc)
            self.last_daily_basic_reason = str(exc)
            return {}
        if frame is None or frame.empty:
            self.last_daily_basic_status = "symbol_missing"
            self.last_daily_basic_reason = ""
            return {}
        row = frame.sort_values("trade_date", ascending=False).iloc[0] if "trade_date" in frame.columns else frame.iloc[0]
        self.last_daily_basic_status = "available"
        self.last_daily_basic_reason = ""
        payload = {
            "pe": _safe_float(row.get("pe")),
            "pb": _safe_float(row.get("pb")),
            "ps": _safe_float(row.get("ps")),
            "dividend_yield": _safe_float(row.get("dv_ratio")),
            "total_mv": _safe_float(row.get("total_mv")),
            "circ_mv": _safe_float(row.get("circ_mv")),
        }
        return {key: value for key, value in payload.items() if value is not None}

    def get_earnings_forecast_snapshot(self, symbol: str, as_of: str) -> ForecastSnapshot:
        ts_code = _normalize_symbol(symbol)
        if not self.allow_live:
            return ForecastSnapshot(
                symbol=ts_code,
                as_of=str(as_of),
                available=False,
                source="neutral",
                provider="none",
                data_quality={
                    "status": "neutral_snapshot",
                    "reason": "provider_missing",
                    "provider_missing": True,
                    "snapshot_missing": False,
                    "missing_scope": "global",
                },
                provenance={"source_priority": "tushare_primary", "provider_missing": True},
                notes=["forecast_provider_missing"],
            )
        for api_name, source in (("report_rc", "tushare_report_rc"), ("forecast", "tushare_forecast")):
            try:
                kwargs = {"ts_code": ts_code}
                if api_name == "report_rc":
                    kwargs["wait_on_quota"] = True
                frame = self._client.query(api_name, **kwargs)
            except Exception as exc:
                self._logger.warning("%s forecast query failed for %s: %s", api_name, symbol, exc)
                continue
            if frame is None or frame.empty:
                continue
            row = frame.iloc[0]
            min_profit = _safe_float(row.get("net_profit_min"))
            max_profit = _safe_float(row.get("net_profit_max"))
            last_profit = _safe_float(row.get("last_parent_net"))
            avg_profit = None
            if min_profit is not None and max_profit is not None:
                avg_profit = (min_profit + max_profit) / 2.0
            elif min_profit is not None:
                avg_profit = min_profit
            eps_growth = _safe_float(row.get("eps_growth")) or 0.0
            profit_growth = 0.0
            if avg_profit is not None and last_profit and last_profit > 0:
                profit_growth = (avg_profit / last_profit) - 1.0
            return ForecastSnapshot(
                symbol=ts_code,
                as_of=str(as_of),
                available=True,
                source=source,
                provider=source,
                eps_growth=float(eps_growth),
                revenue_growth_forecast=float(profit_growth),
                forecast_revision=float(profit_growth),
                coverage_count=int(len(frame)),
                confidence=min(0.9, 0.4 + min(len(frame), 10) * 0.04),
                data_quality={"status": "provider_snapshot", "provider_missing": False, "missing_scope": ""},
                provenance={"source_priority": "tushare_primary", "provider": source},
            )
        return ForecastSnapshot(
            symbol=ts_code,
            as_of=str(as_of),
            available=False,
            source="neutral",
            provider="none",
            data_quality={
                "status": "neutral_snapshot",
                "reason": "snapshot_missing",
                "provider_missing": False,
                "snapshot_missing": True,
                "missing_scope": "symbol",
            },
            provenance={"source_priority": "tushare_primary"},
            notes=["forecast_snapshot_missing"],
        )
