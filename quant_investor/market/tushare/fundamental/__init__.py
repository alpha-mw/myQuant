"""Stable Fundamental planning and offline-testable Tushare acquisition."""

from .acquisition import acquire_fundamental_partitions
from .models import FundamentalAcquisitionError, SOURCE_ENDPOINTS, SOURCE_TABLES
from .official_acquisition import (
    acquire_official_fundamental_partitions,
    validate_official_partition_request_receipt,
)
from .official_plan import (
    build_official_partition_plan,
    replay_official_partition_requests,
    validate_official_partition_plan,
)
from .receipts import (
    build_fundamental_partition_receipt,
    validate_fundamental_partition_receipt,
)
from .schedule import (
    build_fundamental_execution_closure,
    build_fundamental_request_plan,
    validate_fundamental_execution_closure,
    validate_fundamental_request_plan,
)

__all__ = [
    "FundamentalAcquisitionError",
    "SOURCE_ENDPOINTS",
    "SOURCE_TABLES",
    "acquire_fundamental_partitions",
    "acquire_official_fundamental_partitions",
    "build_fundamental_execution_closure",
    "build_fundamental_partition_receipt",
    "build_fundamental_request_plan",
    "build_official_partition_plan",
    "replay_official_partition_requests",
    "validate_fundamental_execution_closure",
    "validate_fundamental_partition_receipt",
    "validate_fundamental_request_plan",
    "validate_official_partition_plan",
    "validate_official_partition_request_receipt",
]
