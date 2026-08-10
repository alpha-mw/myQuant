"""Synthetic-only fixture writer for V17 mainline tests.

This module is not imported by :mod:`quant_investor.v17_mainline` and must not
be used as a production publisher.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Mapping

from .constants import (
    EMPTY_SHA256,
    FORMAL_OUTPUT_SCHEMA_ID,
    PORTFOLIO_OUTPUT_SCHEMA_ID,
    PROTOCOL,
    SOURCE_CLOSURE_SCHEMA_ID,
)
from .contracts import (
    build_active_pointer,
    build_mainline_run,
    build_ref,
    canonical_bytes,
    require_identifier,
    seal_document,
)
from .runtime import active_pointer_path, mainline_run_path
from .storage import MainlineStore, StoredBytes, canonical_relative_path


class _SyntheticFixtureStore(MainlineStore):
    """Test-only escape hatch for synthetic transitive authority bytes."""

    def _validate_write_path(self, value):  # type: ignore[no-untyped-def]
        return canonical_relative_path(value)


@dataclass(frozen=True)
class SyntheticFixture:
    strategy_id: str
    run_id: str
    pointer_path: str
    pointer_sha256: str
    run_path: str
    run_sha256: str
    formal_path: str
    portfolio_path: str
    source_closure_path: str


def write_synthetic_governed_bytes_for_tests(
    workspace_root: str | Path,
    *,
    relative_path: str,
    raw: bytes,
    synthetic_only: bool = False,
) -> StoredBytes:
    """Write exact fixture bytes beneath any read-governed root in tests only."""

    if synthetic_only is not True:
        raise ValueError("synthetic governed writer requires synthetic_only=True")
    return _SyntheticFixtureStore(workspace_root).write_exact_once(relative_path, raw)


def write_synthetic_fixture_for_tests(
    workspace_root: str | Path,
    *,
    strategy_id: str = "cn-mainline",
    run_id: str = "synthetic-run-1",
    timestamp: str = "2026-08-04T00:00:00Z",
    expected_pointer_sha256: str = EMPTY_SHA256,
    run_overrides: Mapping[str, Any] | None = None,
    formal_overrides: Mapping[str, Any] | None = None,
    portfolio_overrides: Mapping[str, Any] | None = None,
    source_overrides: Mapping[str, Any] | None = None,
    additional_formal_evidence_refs: Sequence[Mapping[str, Any]] = (),
    synthetic_only: bool = False,
    allow_invalid_run_for_tests: bool = False,
) -> SyntheticFixture:
    """Write a fully closed synthetic fixture through exact-byte storage.

    ``synthetic_only=True`` is deliberately mandatory so callers cannot mistake
    this low-level test support for an operational publication surface.
    """

    if synthetic_only is not True:
        raise ValueError("synthetic fixture writer requires synthetic_only=True")
    strategy = require_identifier(strategy_id, label="strategy_id")
    run = require_identifier(run_id, label="run_id")
    store = _SyntheticFixtureStore(workspace_root)

    portfolio_path = (
        f"results/v17_v4_formal_research/strategies/{strategy}/" f"synthetic/{run}/portfolio.json"
    )
    portfolio_body: dict[str, Any] = {
        "version": PORTFOLIO_OUTPUT_SCHEMA_ID,
        "protocol_version": PROTOCOL,
        "strategy_id": strategy,
        "run_id": run,
        "status": "COMPLETE",
        "cash_weight": "0.60",
        "gross_weight": "0.40",
        "targets": [
            {
                "symbol": "000001.SZ",
                "current_target": "0.10",
                "final_target": "0.15",
                "lane": "SELECTION_POOL",
            },
            {
                "symbol": "600000.SH",
                "current_target": "0.25",
                "final_target": "0.25",
                "lane": "SELECTION_POOL",
            },
        ],
    }
    if portfolio_overrides:
        portfolio_body.update(dict(portfolio_overrides))
    portfolio_document = seal_document(portfolio_body)
    portfolio_raw = canonical_bytes(portfolio_document)
    portfolio_ref = build_ref(
        schema_id=PORTFOLIO_OUTPUT_SCHEMA_ID,
        relative_path=portfolio_path,
        raw=portfolio_raw,
    )

    formal_path = (
        f"results/v17_v4_formal_research/strategies/{strategy}/" f"synthetic/{run}/formal.json"
    )
    formal_body: dict[str, Any] = {
        "version": FORMAL_OUTPUT_SCHEMA_ID,
        "protocol_version": PROTOCOL,
        "strategy_id": strategy,
        "terminal_state": "PUBLISHED_RESEARCH_ONLY",
        "authority": {
            "broker": False,
            "execution": False,
            "formal_research_publication": True,
            "order": False,
            "research_runtime_default": True,
            "trade": False,
        },
        "evidence_refs": [
            dict(portfolio_ref),
            *(dict(reference) for reference in additional_formal_evidence_refs),
        ],
    }
    if formal_overrides:
        formal_body.update(dict(formal_overrides))
    formal_document = seal_document(formal_body)
    formal_raw = canonical_bytes(formal_document)
    formal_ref = build_ref(
        schema_id=FORMAL_OUTPUT_SCHEMA_ID,
        relative_path=formal_path,
        raw=formal_raw,
    )

    source_path = "data/private/v17_v4_sources/pit_catalog/generations/" f"synthetic-{run}.json"
    source_body: dict[str, Any] = {
        "version": SOURCE_CLOSURE_SCHEMA_ID,
        "protocol_version": PROTOCOL,
        "strategy_id": strategy,
        "source_closure_sha256": "7" * 64,
    }
    if source_overrides:
        source_body.update(dict(source_overrides))
    source_document = seal_document(source_body)
    source_raw = canonical_bytes(source_document)
    source_ref = build_ref(
        schema_id=SOURCE_CLOSURE_SCHEMA_ID,
        relative_path=source_path,
        raw=source_raw,
    )

    run_document = build_mainline_run(
        canonical_strategy_id=strategy,
        run_id=run,
        created_at=timestamp,
        formal_output_ref=formal_ref,
        portfolio_output_ref=portfolio_ref,
        source_closure_ref=source_ref,
    )
    if run_overrides:
        body = dict(run_document)
        body.pop("semantic_sha256", None)
        body.update(dict(run_overrides))
        run_document = seal_document(body)
    if not allow_invalid_run_for_tests:
        from .contracts import validate_mainline_run

        validate_mainline_run(run_document)
    run_raw = canonical_bytes(run_document)
    run_path = mainline_run_path(strategy, run)
    run_ref = build_ref(
        schema_id="myquant.v17.v4.mainline-run.v1",
        relative_path=run_path,
        raw=run_raw,
    )
    pointer_document = build_active_pointer(
        canonical_strategy_id=strategy,
        run_id=run,
        updated_at=timestamp,
        run_ref=run_ref,
    )
    pointer_raw = canonical_bytes(pointer_document)
    pointer_path = active_pointer_path(strategy)

    # Construct every byte before the first filesystem mutation.
    store.write_exact_once(portfolio_path, portfolio_raw)
    store.write_exact_once(formal_path, formal_raw)
    store.write_exact_once(source_path, source_raw)
    stored_run = store.write_exact_once(run_path, run_raw)
    stored_pointer = store.compare_and_swap(
        pointer_path,
        pointer_raw,
        expected_sha256=expected_pointer_sha256,
    )
    return SyntheticFixture(
        strategy_id=strategy,
        run_id=run,
        pointer_path=pointer_path,
        pointer_sha256=stored_pointer.byte_sha256,
        run_path=run_path,
        run_sha256=stored_run.byte_sha256,
        formal_path=formal_path,
        portfolio_path=portfolio_path,
        source_closure_path=source_path,
    )


__all__ = [
    "SyntheticFixture",
    "write_synthetic_fixture_for_tests",
    "write_synthetic_governed_bytes_for_tests",
]
