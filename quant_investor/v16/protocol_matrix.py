"""Frozen public protocol matrix for the research-only v16 candidate runtime."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Mapping

ARCHITECTURE_VERSION = "16.0.0"
BRANCH_VERSION = "v16.four-branch"
LIKELIHOOD_VERSION = "v16.four-evidence"
OUTCOME_VERSION = "v16.four-evidence"
CALIBRATION_VERSION = "v16.four-evidence"
POSTERIOR_VERSION = "v16.four-evidence"
IC_VERSION = "v16.codex-authoritative"
HANDOFF_VERSION = "v1"
ELIGIBILITY_VERSION = "v1"
RISK_ADVISOR_VERSION = "v1"
REPORT_VERSION = "v16"
READINESS_VERSION = "v16"
DASHBOARD_VERSION = "v16"
FACTOR_GOVERNANCE_VERSION = "v4"
RESULTS_SUBDIRECTORY = "results/v16"


PROTOCOL_VERSIONS: Mapping[str, str] = MappingProxyType(
    {
        "architecture_version": ARCHITECTURE_VERSION,
        "branch_version": BRANCH_VERSION,
        "likelihood_version": LIKELIHOOD_VERSION,
        "outcome_version": OUTCOME_VERSION,
        "calibration_version": CALIBRATION_VERSION,
        "posterior_version": POSTERIOR_VERSION,
        "ic_version": IC_VERSION,
        "handoff_version": HANDOFF_VERSION,
        "eligibility_version": ELIGIBILITY_VERSION,
        "risk_advisor_version": RISK_ADVISOR_VERSION,
        "report_version": REPORT_VERSION,
        "readiness_version": READINESS_VERSION,
        "dashboard_version": DASHBOARD_VERSION,
        "factor_governance_version": FACTOR_GOVERNANCE_VERSION,
    }
)


def protocol_envelope() -> dict[str, str]:
    """Return a mutable serialization copy of the exact protocol matrix."""

    return dict(PROTOCOL_VERSIONS)


def require_exact_v16_protocol(payload: Mapping[str, object]) -> None:
    """Reject missing, unknown, or legacy protocol fields before data parsing."""

    actual_keys = set(payload)
    expected_keys = set(PROTOCOL_VERSIONS)
    missing = sorted(expected_keys - actual_keys)
    unknown = sorted(actual_keys - expected_keys)
    if missing or unknown:
        raise ValueError(
            f"v16 protocol envelope keys mismatch: missing={missing}, unknown={unknown}"
        )
    mismatches = {
        key: {"expected": expected, "actual": payload[key]}
        for key, expected in PROTOCOL_VERSIONS.items()
        if payload[key] != expected
    }
    if mismatches:
        raise ValueError(f"legacy or mismatched protocol versions are not accepted: {mismatches}")


def results_v16_root(repo_root: str | Path) -> Path:
    """Resolve the dedicated output root without creating or mutating it."""

    return Path(repo_root).resolve() / RESULTS_SUBDIRECTORY
