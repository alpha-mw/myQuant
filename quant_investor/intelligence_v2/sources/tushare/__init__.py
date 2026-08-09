"""Governed, strict-Decimal Tushare capability contracts."""

from .contracts import (
    build_endpoint_execution_plan,
    build_tushare_capability_receipt,
    build_tushare_endpoint_policy,
    build_tushare_execution_receipt,
    build_tushare_request_receipt,
    validate_endpoint_execution_plan,
    validate_tushare_capability_receipt,
    validate_tushare_endpoint_policy,
    validate_tushare_execution_receipt,
    validate_tushare_request_receipt,
)
from .models import TushareContractError, TushareRequestClient
from .probe import probe_tushare_capabilities
from .industry import compile_tushare_sw2021_industry_source

__all__ = [
    "TushareContractError",
    "TushareRequestClient",
    "build_endpoint_execution_plan",
    "build_tushare_capability_receipt",
    "build_tushare_endpoint_policy",
    "build_tushare_execution_receipt",
    "build_tushare_request_receipt",
    "compile_tushare_sw2021_industry_source",
    "probe_tushare_capabilities",
    "validate_endpoint_execution_plan",
    "validate_tushare_capability_receipt",
    "validate_tushare_endpoint_policy",
    "validate_tushare_execution_receipt",
    "validate_tushare_request_receipt",
]
