"""Build one immutable experiment registration per daily forward run."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from .._artifacts import NO_AUTHORITY, PROTOCOL_VERSION, seal, session, timestamp
from ..label_maturity import LABEL_HORIZONS

VERSION: Final = "myquant.v17.v4.research-experiment-registry.v1"
VARIANTS: Final = (
    "v17-quant-core",
    "v17-quant-plus-industry",
    "v17-quant-plus-industry-theme",
)


def build_experiment_registry(
    *,
    experiment_id: str,
    run_id: str,
    strategy_id: str,
    decision_session: str,
    cutoff: str,
    created_at: str,
    forward_manifest_ref: Mapping[str, Any],
) -> dict[str, Any]:
    document = {
        "authority": dict(NO_AUTHORITY),
        "created_at": timestamp(created_at, label="created_at"),
        "cutoff": timestamp(cutoff, label="cutoff"),
        "decision_session": session(decision_session, label="decision_session"),
        "diagnostic_only": True,
        "experiment_id": experiment_id,
        "factor_governance_write": False,
        "forward_manifest_ref": dict(forward_manifest_ref),
        "historical_backfill_eligible": False,
        "label_horizons": list(LABEL_HORIZONS),
        "production_governance_eligible": False,
        "protocol_version": PROTOCOL_VERSION,
        "research_only": True,
        "run_id": run_id,
        "strategy_id": strategy_id,
        "variants": list(VARIANTS),
        "version": VERSION,
    }
    return seal(document, identity_field="registry_id")


__all__ = ["VARIANTS", "VERSION", "build_experiment_registry"]
