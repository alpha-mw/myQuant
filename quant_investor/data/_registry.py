"""Provider endpoint registry with backwards-compatible Tushare metadata.

The generic catalog is descriptive only.  Registering a provider here does not
grant fetch, publication, pointer-mutation, or production authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from quant_investor.config import Config


@dataclass(frozen=True)
class TushareEndpointSpec:
    name: str
    rate_limit_per_min: int
    source_priority: str = "tushare_primary"
    point_in_time: bool = True


ProviderRole = Literal["production_primary", "primary_candidate", "validator"]


@dataclass(frozen=True)
class ProviderEndpointSpec:
    """Provider-neutral endpoint metadata used by migration tooling."""

    provider: str
    name: str
    rate_limit_per_min: int | None
    role: ProviderRole
    point_in_time: bool = True
    activation_authorized: bool = False


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


# RQData remains a shadow candidate until an exact generation passes the
# separately governed reconciliation and promotion path.  These endpoint names
# follow the official RQData Python/HTTP API surface but cause no import or I/O.
RQDATA_CATALOG: dict[str, ProviderEndpointSpec] = {
    "get_price": ProviderEndpointSpec(
        provider="rqdata",
        name="get_price",
        rate_limit_per_min=None,
        role="primary_candidate",
    ),
    "all_instruments": ProviderEndpointSpec(
        provider="rqdata",
        name="all_instruments",
        rate_limit_per_min=None,
        role="primary_candidate",
    ),
    "get_trading_dates": ProviderEndpointSpec(
        provider="rqdata",
        name="get_trading_dates",
        rate_limit_per_min=None,
        role="primary_candidate",
    ),
    "get_pit_financials": ProviderEndpointSpec(
        provider="rqdata",
        name="get_pit_financials",
        rate_limit_per_min=None,
        role="primary_candidate",
    ),
}


def get_endpoint_spec(api_name: str) -> TushareEndpointSpec:
    return TUSHARE_CATALOG.get(str(api_name), TushareEndpointSpec(str(api_name), _RATE))


def get_provider_endpoint_spec(provider: str, api_name: str) -> ProviderEndpointSpec:
    """Return exact provider-neutral metadata without silently changing source."""

    normalized_provider = str(provider or "").strip().lower()
    normalized_name = str(api_name or "").strip()
    if not normalized_provider:
        raise ValueError("provider is required")
    if not normalized_name:
        raise ValueError("api_name is required")
    if normalized_provider == "tushare":
        try:
            legacy = TUSHARE_CATALOG[normalized_name]
        except KeyError as exc:
            raise KeyError(f"unregistered tushare endpoint: {normalized_name}") from exc
        return ProviderEndpointSpec(
            provider="tushare",
            name=legacy.name,
            rate_limit_per_min=legacy.rate_limit_per_min,
            role="production_primary",
            point_in_time=legacy.point_in_time,
        )
    if normalized_provider == "rqdata":
        try:
            return RQDATA_CATALOG[normalized_name]
        except KeyError as exc:
            raise KeyError(f"unregistered rqdata endpoint: {normalized_name}") from exc
    raise KeyError(f"unregistered provider: {normalized_provider}")
