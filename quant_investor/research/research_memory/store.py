"""Build immutable, content-addressed daily research memory entries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from .._artifacts import NO_AUTHORITY, PROTOCOL_VERSION, seal, session, sorted_refs, timestamp

VERSION: Final = "myquant.v17.v4.research-memory-entry.v1"


def build_memory_entry(
    *,
    run_id: str,
    strategy_id: str,
    decision_session: str,
    cutoff: str,
    created_at: str,
    source_refs: Sequence[Mapping[str, Any]],
    run_state: str,
    limitation_codes: Sequence[str],
) -> dict[str, Any]:
    document = {
        "authority": dict(NO_AUTHORITY),
        "created_at": timestamp(created_at, label="created_at"),
        "cutoff": timestamp(cutoff, label="cutoff"),
        "decision_session": session(decision_session, label="decision_session"),
        "diagnostic_only": True,
        "factor_governance_write": False,
        "historical_backfill_eligible": False,
        "limitation_codes": sorted(set(limitation_codes), key=lambda value: value.encode("ascii")),
        "production_governance_eligible": False,
        "protocol_version": PROTOCOL_VERSION,
        "research_only": True,
        "run_id": run_id,
        "run_state": run_state,
        "source_refs": sorted_refs(list(source_refs)),
        "strategy_id": strategy_id,
        "version": VERSION,
    }
    return seal(document, identity_field="entry_id")


__all__ = ["VERSION", "build_memory_entry"]
