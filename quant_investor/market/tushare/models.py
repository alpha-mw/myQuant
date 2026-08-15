"""Stable models and semantic kinds for governed Tushare acquisition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final, Protocol

from quant_investor.market.tushare_transport import TushareResponse

from ._core import TushareDataContractError

ENDPOINT_POLICY_KIND: Final = "market.tushare.endpoint_policy"
EXECUTION_PLAN_KIND: Final = "market.tushare.endpoint_execution_plan"
CAPABILITY_RECEIPT_KIND: Final = "market.tushare.capability_receipt"
REQUEST_RECEIPT_KIND: Final = "market.tushare.request_receipt"
EXECUTION_RECEIPT_KIND: Final = "market.tushare.execution_receipt"
SCHEMA_DIAGNOSTIC_RECEIPT_KIND: Final = "market.tushare.schema_diagnostic_receipt"

PERMISSION_CLASSES: Final = frozenset({"POINTS", "SEPARATE"})
CAPABILITY_STATUSES: Final = frozenset(
    {
        "NOT_PROBED",
        "AVAILABLE",
        "EMPTY",
        "PROVIDER_ERROR",
        "SCHEMA_MISMATCH",
        "INCOMPLETE",
        "TRANSPORT_ERROR",
    }
)
LANES: Final = frozenset({"FUNDAMENTAL", "INDUSTRY", "THEME", "DIAGNOSTIC"})
INCOMPLETE_BLOCKERS: Final = frozenset(
    {
        "ROW_LIMIT_HIT",
        "HAS_MORE",
        "KEYSET_MISSING",
        "SCOPE_MISMATCH",
        "COVERAGE_INCOMPLETE",
        "COUNT_MISMATCH",
        "DUPLICATE_ROWS",
    }
)


class TushareContractError(TushareDataContractError):
    """Fail-closed stable Tushare source contract error."""


class TushareRequestClient(Protocol):
    """The only transport operation admitted by acquisition code."""

    def request(
        self,
        *,
        api_name: str,
        params: Mapping[str, Any],
        expected_fields: Sequence[str],
    ) -> TushareResponse: ...


__all__ = [
    "CAPABILITY_RECEIPT_KIND",
    "CAPABILITY_STATUSES",
    "ENDPOINT_POLICY_KIND",
    "EXECUTION_PLAN_KIND",
    "EXECUTION_RECEIPT_KIND",
    "INCOMPLETE_BLOCKERS",
    "LANES",
    "PERMISSION_CLASSES",
    "REQUEST_RECEIPT_KIND",
    "SCHEMA_DIAGNOSTIC_RECEIPT_KIND",
    "TushareContractError",
    "TushareRequestClient",
]
