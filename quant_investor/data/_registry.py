"""Tushare endpoint registry with source-priority metadata."""

from __future__ import annotations

from dataclasses import dataclass

from quant_investor.config import Config


@dataclass(frozen=True)
class TushareEndpointSpec:
    name: str
    rate_limit_per_min: int
    source_priority: str = "tushare_primary"
    point_in_time: bool = True


def _default_rate() -> int:
    try:
        return int(Config.TUSHARE_RATE_LIMIT_PER_MIN)
    except Exception:
        return 500


_RATE = _default_rate()

TUSHARE_CATALOG: dict[str, TushareEndpointSpec] = {
    "daily": TushareEndpointSpec("daily", _RATE),
    "adj_factor": TushareEndpointSpec("adj_factor", _RATE),
    "daily_basic": TushareEndpointSpec("daily_basic", _RATE),
    "suspend_d": TushareEndpointSpec("suspend_d", _RATE),
    "limit_list_d": TushareEndpointSpec("limit_list_d", _RATE),
    "trade_cal": TushareEndpointSpec("trade_cal", _RATE),
    "fina_indicator": TushareEndpointSpec("fina_indicator", _RATE),
    "income": TushareEndpointSpec("income", _RATE),
    "balancesheet": TushareEndpointSpec("balancesheet", _RATE),
    "cashflow": TushareEndpointSpec("cashflow", _RATE),
    "forecast": TushareEndpointSpec("forecast", _RATE),
    "report_rc": TushareEndpointSpec("report_rc", 2),
    "moneyflow": TushareEndpointSpec("moneyflow", _RATE),
    "margin": TushareEndpointSpec("margin", _RATE),
    "hsgt_top10": TushareEndpointSpec("hsgt_top10", _RATE),
    "stk_factor": TushareEndpointSpec("stk_factor", _RATE),
    "us_daily": TushareEndpointSpec("us_daily", _RATE),
    "us_fina_indicator": TushareEndpointSpec("us_fina_indicator", _RATE),
    "us_income": TushareEndpointSpec("us_income", _RATE),
    "us_balancesheet": TushareEndpointSpec("us_balancesheet", _RATE),
    "us_cashflow": TushareEndpointSpec("us_cashflow", _RATE),
}


def get_endpoint_spec(api_name: str) -> TushareEndpointSpec:
    return TUSHARE_CATALOG.get(str(api_name), TushareEndpointSpec(str(api_name), _RATE))
