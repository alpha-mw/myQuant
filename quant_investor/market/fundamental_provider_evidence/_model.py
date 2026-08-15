"""Private constants and error translation for provider-evidence replay."""

from __future__ import annotations

from functools import wraps
from typing import Callable, Final, ParamSpec, TypeVar

from ._codec import FundamentalProviderEvidenceError

# These strings are frozen byte-level discriminators in already-persisted
# provider evidence. Stable callers use semantic function names and never
# branch on these private constants directly.
ENDPOINT_EXECUTION_PLAN_SCHEMA: Final = (
    "myquant.v17.intelligence-v2.tushare-endpoint-execution-plan.v1"
)
PROVIDER_MANIFEST_SCHEMA: Final = "cn-fundamental-provider-manifest.v4"
REQUEST_PLAN_SCHEMA: Final = "myquant.v17.fundamental-request-plan.v4"
EXECUTION_CLOSURE_SCHEMA: Final = "myquant.v17.fundamental-execution-closure.v4"
PHYSICAL_REQUEST_RECEIPT_SCHEMA: Final = "myquant.v17.provider-physical-request-receipt.v4"
LOGICAL_COVERAGE_SCHEMA: Final = "myquant.v17.logical-symbol-table-coverage.v4"
RAW_TABLE_EVIDENCE_SCHEMA: Final = "myquant.v17.raw-table-evidence.v4"
LEGACY_COMPARISON_POLICY_SCHEMA: Final = "myquant.v17.fundamental-comparison-policy.v2"
COMPARISON_POLICY_SCHEMA: Final = "myquant.v17.fundamental-comparison-policy.v3"
RECONCILIATION_RECEIPT_SCHEMA: Final = "myquant.v17.fundamental-reconciliation-receipt.v1"
PROVIDER_EVIDENCE_FILESET_SCHEMA: Final = "myquant.v17.provider-evidence-fileset.v1"

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
ENDPOINT_LANES: Final = frozenset({"FUNDAMENTAL", "INDUSTRY", "THEME", "DIAGNOSTIC"})
PERMISSION_CLASSES: Final = frozenset({"POINTS", "SEPARATE"})
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


_P = ParamSpec("_P")
_R = TypeVar("_R")


def provider_evidence_contract(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """Keep all persisted-evidence failures in one stable error domain."""

    @wraps(function)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        try:
            return function(*args, **kwargs)
        except FundamentalProviderEvidenceError:
            raise
        except (TypeError, ValueError) as exc:
            raise FundamentalProviderEvidenceError(str(exc)) from exc

    return wrapped


__all__ = ["FundamentalProviderEvidenceError"]
