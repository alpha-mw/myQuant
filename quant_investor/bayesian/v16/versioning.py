"""Bayesian aliases into the frozen research-only v16 protocol matrix."""

from __future__ import annotations


from quant_investor.v16.protocol_matrix import (
    ARCHITECTURE_VERSION,
    BRANCH_VERSION as BRANCH_SCHEMA_VERSION,
    CALIBRATION_VERSION as CALIBRATION_SCHEMA_VERSION,
    LIKELIHOOD_VERSION as LIKELIHOOD_SCHEMA_VERSION,
    POSTERIOR_VERSION as POSTERIOR_SCHEMA_VERSION,
)

PRIOR_SCHEMA_VERSION = "v16.base-rate"


def output_version_payload() -> dict[str, str]:
    return {
        "architecture_version": ARCHITECTURE_VERSION,
        "branch_schema_version": BRANCH_SCHEMA_VERSION,
        "prior_schema_version": PRIOR_SCHEMA_VERSION,
        "likelihood_schema_version": LIKELIHOOD_SCHEMA_VERSION,
        "posterior_schema_version": POSTERIOR_SCHEMA_VERSION,
        "calibration_schema_version": CALIBRATION_SCHEMA_VERSION,
    }


__all__ = [
    "ARCHITECTURE_VERSION",
    "BRANCH_SCHEMA_VERSION",
    "CALIBRATION_SCHEMA_VERSION",
    "LIKELIHOOD_SCHEMA_VERSION",
    "POSTERIOR_SCHEMA_VERSION",
    "PRIOR_SCHEMA_VERSION",
    "output_version_payload",
]
