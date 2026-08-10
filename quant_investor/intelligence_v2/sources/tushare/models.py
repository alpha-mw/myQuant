"""Models and constants for the governed Tushare v2 source seam."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final, Protocol

from ..._core import IntelligenceV2ContractError
from ....v17_v4_runtime.tushare_https import TushareResponse

ENDPOINT_POLICY_VERSION: Final = "myquant.v17.intelligence-v2.tushare-endpoint-policy.v1"
EXECUTION_PLAN_VERSION: Final = "myquant.v17.intelligence-v2.tushare-endpoint-execution-plan.v1"
CAPABILITY_RECEIPT_VERSION: Final = "myquant.v17.intelligence-v2.tushare-capability-receipt.v1"
REQUEST_RECEIPT_VERSION: Final = "myquant.v17.intelligence-v2.tushare-request-receipt.v1"
EXECUTION_RECEIPT_VERSION: Final = "myquant.v17.intelligence-v2.tushare-execution-receipt.v1"
SCHEMA_DIAGNOSTIC_RECEIPT_VERSION: Final = (
    "myquant.v17.intelligence-v2.tushare-schema-diagnostic-receipt.v1"
)

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


class TushareContractError(IntelligenceV2ContractError):
    """Fail-closed v2 Tushare source contract error."""


class TushareRequestClient(Protocol):
    """The only transport operation admitted by the capability probe."""

    def request(
        self,
        *,
        api_name: str,
        params: Mapping[str, Any],
        expected_fields: Sequence[str],
    ) -> TushareResponse: ...


__all__ = [
    "CAPABILITY_RECEIPT_VERSION",
    "CAPABILITY_STATUSES",
    "ENDPOINT_POLICY_VERSION",
    "EXECUTION_PLAN_VERSION",
    "EXECUTION_RECEIPT_VERSION",
    "INCOMPLETE_BLOCKERS",
    "LANES",
    "PERMISSION_CLASSES",
    "REQUEST_RECEIPT_VERSION",
    "SCHEMA_DIAGNOSTIC_RECEIPT_VERSION",
    "TushareContractError",
    "TushareRequestClient",
]
