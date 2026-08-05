"""Availability-aware research branch fusion."""

from .branches import build_fundamental_branch, build_quant_branch, validate_branch
from .engine import (
    FUSION_RECEIPT_VERSION,
    fuse_research_branches,
    validate_fusion_receipt,
)

__all__ = [
    "FUSION_RECEIPT_VERSION",
    "build_fundamental_branch",
    "build_quant_branch",
    "fuse_research_branches",
    "validate_branch",
    "validate_fusion_receipt",
]
