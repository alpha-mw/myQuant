"""The production evidence blocker must not punish perfect exposure evidence.

``float(value or default)`` treats a legitimate 0.0 as missing, so a factor
exposure source that reconstructs *none* of its market caps - the best possible
result - was read as having reconstructed all of them.
"""

from __future__ import annotations

from typing import Any

import pytest

from quant_investor.factors.exposure_maps import (
    GOVERNED_EXPOSURE_SOURCE,
    GOVERNED_SIZE_POLICY,
)
from scripts.mine_quant_branch_factors import _production_market_evidence_blocker


def _exposure(**overrides: Any) -> dict[str, Any]:
    payload = {
        "status": "ready",
        "source": GOVERNED_EXPOSURE_SOURCE,
        "size_policy": GOVERNED_SIZE_POLICY,
        "catalog_validated": True,
        "coverage_ratio": 0.9568,
        "evaluation_date_coverage_ratio": 1.0,
        "min_cross_section_coverage_ratio": 0.9972,
        "combined_size_pair_coverage_ratio": 0.9979,
        "pit_size_pair_coverage_ratio": 0.9979,
        "reconstructed_size_pair_ratio": 0.0,
        "share_reference_covers_evaluation_end": True,
        "sector_count": 110,
        "size_bucket_count": 3,
    }
    payload.update(overrides)
    return payload


def _evidence(**exposure_overrides: Any) -> dict[str, Any]:
    return {
        "backend": "parquet",
        "mode_policy": "strict",
        "pointer_status": "OK",
        "snapshot_id": "20260804T170834Z",
        "coverage_complete": True,
        "table_root_exists": True,
        "serving_root_exists": True,
        "manifest_exists": True,
        "expected_symbol_count": 5502,
        "loaded_symbol_count": 5735,
        "factor_exposure_evidence": _exposure(**exposure_overrides),
    }


def test_zero_reconstruction_is_the_best_result_not_a_blocker() -> None:
    blocker = _production_market_evidence_blocker(
        universes=["full_a"],
        market_evidence=_evidence(reconstructed_size_pair_ratio=0.0),
    )

    assert blocker == ""


def test_reconstruction_above_the_ceiling_still_blocks() -> None:
    blocker = _production_market_evidence_blocker(
        universes=["full_a"],
        market_evidence=_evidence(reconstructed_size_pair_ratio=0.36),
    )

    assert blocker == "factor_exposure_reconstruction_above_35pct"


def test_missing_reconstruction_evidence_blocks() -> None:
    evidence = _evidence()
    evidence["factor_exposure_evidence"].pop("reconstructed_size_pair_ratio")

    blocker = _production_market_evidence_blocker(
        universes=["full_a"], market_evidence=evidence
    )

    assert blocker == "factor_exposure_reconstruction_above_35pct"


def test_governed_exposure_source_and_size_policy_are_accepted() -> None:
    assert (
        _production_market_evidence_blocker(
            universes=["full_a"], market_evidence=_evidence()
        )
        == ""
    )


def test_legacy_hybrid_exposure_source_is_still_accepted() -> None:
    blocker = _production_market_evidence_blocker(
        universes=["full_a"],
        market_evidence=_evidence(
            source="strict_parquet_hybrid_market_cap_exposure",
            size_policy=(
                "same_trade_date_total_mv_then_asof_total_share_times_close"
            ),
            reconstructed_size_pair_ratio=0.27,
        ),
    )

    assert blocker == ""


def test_unknown_exposure_source_blocks() -> None:
    blocker = _production_market_evidence_blocker(
        universes=["full_a"],
        market_evidence=_evidence(source="some_unreviewed_source"),
    )

    assert blocker == "factor_exposure_source_not_strict_parquet"


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"status": "blocked"}, "factor_exposure_evidence_not_ready"),
        ({"catalog_validated": False}, "factor_exposure_catalog_not_validated"),
        ({"coverage_ratio": 0.94}, "factor_exposure_coverage_below_95pct"),
        (
            {"share_reference_covers_evaluation_end": False},
            "factor_exposure_share_reference_stale_for_evaluation",
        ),
        ({"size_bucket_count": 2}, "factor_exposure_size_bucket_count_below_3"),
    ],
)
def test_exposure_shortfalls_still_block(
    override: dict[str, Any], expected: str
) -> None:
    assert (
        _production_market_evidence_blocker(
            universes=["full_a"], market_evidence=_evidence(**override)
        )
        == expected
    )
