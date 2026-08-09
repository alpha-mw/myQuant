"""Versioned constants for Fundamental bulk-provider evidence v4."""

from __future__ import annotations

from functools import wraps
from typing import Callable, Final, ParamSpec, TypeVar

from ...._core import IntelligenceV2ContractError

PROVIDER_MANIFEST_V4: Final = "cn-fundamental-provider-manifest.v4"
REQUEST_PLAN_V4: Final = "myquant.v17.fundamental-request-plan.v4"
EXECUTION_CLOSURE_V4: Final = "myquant.v17.fundamental-execution-closure.v4"
PHYSICAL_REQUEST_RECEIPT_V4: Final = "myquant.v17.provider-physical-request-receipt.v4"
LOGICAL_COVERAGE_V4: Final = "myquant.v17.logical-symbol-table-coverage.v4"
RAW_TABLE_EVIDENCE_V4: Final = "myquant.v17.raw-table-evidence.v4"
COMPARISON_POLICY_V1: Final = "myquant.v17.fundamental-comparison-policy.v1"
RECONCILIATION_RECEIPT_V1: Final = "myquant.v17.fundamental-reconciliation-receipt.v1"
PROVIDER_EVIDENCE_FILESET_V1: Final = "myquant.v17.provider-evidence-fileset.v1"

FINANCIAL_ENDPOINTS: Final = {
    "balancesheet": "balancesheet_vip",
    "cashflow": "cashflow_vip",
    "fina_indicator": "fina_indicator_vip",
    "forecast": "forecast_vip",
    "income": "income_vip",
}
SOURCE_ENDPOINTS: Final = {**FINANCIAL_ENDPOINTS, "daily_basic": "daily_basic"}
SOURCE_TABLES: Final = tuple(sorted(SOURCE_ENDPOINTS))
LANES: Final = frozenset({"BASELINE", "VIP"})
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
LOGICAL_STATUSES: Final = frozenset({"COMPLETE", "INCOMPLETE"})
SCALAR_KINDS: Final = frozenset({"DATE", "DECIMAL", "INTEGER", "TEXT"})


class FundamentalV4ContractError(IntelligenceV2ContractError):
    """Fail-closed bulk Fundamental provider contract error."""


_P = ParamSpec("_P")
_R = TypeVar("_R")


def fundamental_v4_contract(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """Translate shared v2 primitive failures into the v4 domain."""

    @wraps(function)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        try:
            return function(*args, **kwargs)
        except FundamentalV4ContractError:
            raise
        except IntelligenceV2ContractError as exc:
            raise FundamentalV4ContractError(str(exc)) from exc

    return wrapped


__all__ = [
    "COMPARISON_POLICY_V1",
    "FINANCIAL_ENDPOINTS",
    "FundamentalV4ContractError",
    "LANES",
    "LOGICAL_COVERAGE_V4",
    "LOGICAL_STATUSES",
    "PHYSICAL_REQUEST_RECEIPT_V4",
    "PHYSICAL_STATUSES",
    "PROVIDER_MANIFEST_V4",
    "RAW_TABLE_EVIDENCE_V4",
    "RECONCILIATION_RECEIPT_V1",
    "REQUEST_PLAN_V4",
    "SCALAR_KINDS",
    "SOURCE_ENDPOINTS",
    "SOURCE_TABLES",
    "fundamental_v4_contract",
]
