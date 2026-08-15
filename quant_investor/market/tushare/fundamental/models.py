"""Stable Fundamental acquisition kinds and error boundary."""

from __future__ import annotations

from functools import wraps
from typing import Callable, Final, ParamSpec, TypeVar

from .._core import TushareDataContractError

REQUEST_PLAN_KIND: Final = "market.tushare.fundamental_request_plan"
EXECUTION_CLOSURE_KIND: Final = "market.tushare.fundamental_execution_closure"
PHYSICAL_RECEIPT_KIND: Final = "market.tushare.fundamental_partition_receipt"
OFFICIAL_PARTITION_PLAN_KIND: Final = "market.tushare.fundamental_official_partition_plan"
OFFICIAL_PARTITION_RECEIPT_KIND: Final = "market.tushare.fundamental_official_partition_receipt"

FINANCIAL_ENDPOINTS: Final = {
    "balancesheet": "balancesheet_vip",
    "cashflow": "cashflow_vip",
    "fina_indicator": "fina_indicator_vip",
    "forecast": "forecast_vip",
    "income": "income_vip",
}
SOURCE_ENDPOINTS: Final = {**FINANCIAL_ENDPOINTS, "daily_basic": "daily_basic"}
SOURCE_TABLES: Final = tuple(sorted(SOURCE_ENDPOINTS))
PHYSICAL_STATUSES: Final = frozenset(
    {
        "AVAILABLE",
        "EMPTY",
        "PROVIDER_ERROR",
        "SCHEMA_MISMATCH",
        "INCOMPLETE",
        "TRANSPORT_ERROR",
    }
)


class FundamentalAcquisitionError(TushareDataContractError):
    """Fail-closed error for stable Fundamental acquisition."""

    code = "FUNDAMENTAL_ACQUISITION_INVALID"


_P = ParamSpec("_P")
_R = TypeVar("_R")


def fundamental_contract(function: Callable[_P, _R]) -> Callable[_P, _R]:
    @wraps(function)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        try:
            return function(*args, **kwargs)
        except FundamentalAcquisitionError:
            raise
        except (TypeError, ValueError) as exc:
            raise FundamentalAcquisitionError(str(exc)) from exc

    return wrapped


__all__ = [
    "EXECUTION_CLOSURE_KIND",
    "FINANCIAL_ENDPOINTS",
    "FundamentalAcquisitionError",
    "OFFICIAL_PARTITION_PLAN_KIND",
    "OFFICIAL_PARTITION_RECEIPT_KIND",
    "PHYSICAL_RECEIPT_KIND",
    "PHYSICAL_STATUSES",
    "REQUEST_PLAN_KIND",
    "SOURCE_ENDPOINTS",
    "SOURCE_TABLES",
    "fundamental_contract",
]
