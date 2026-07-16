"""Regression tests for branch-specific source policy boundaries."""

from __future__ import annotations

import pandas as pd

from quant_investor.market.branch_readiness import (
    FUNDAMENTAL_REQUIRED_FIELDS,
    SOURCE_TUSHARE,
    STATUS_BLOCK,
    STATUS_PASS,
    _assess_symbol_records,
    assess_quant_readiness,
)


def test_quant_readiness_keeps_strict_tushare_canonical_policy() -> None:
    frame = pd.DataFrame(
        [
            {
                "trade_date": "20240510",
                "open": 10.0,
                "high": 10.5,
                "low": 9.8,
                "close": 10.2,
                "volume": 1_000.0,
                "amount": 10_200.0,
            }
        ]
    )

    readiness = assess_quant_readiness(
        frames={"000001.SZ": frame},
        symbols=["000001.SZ"],
        as_of="20240510",
    )

    assert readiness.status == STATUS_PASS
    assert readiness.source_priority == SOURCE_TUSHARE
    assert readiness.fallback_used is False
    assert readiness.provider_status == "strict_parquet_snapshot"


def test_fundamental_readiness_still_blocks_non_tushare_primary() -> None:
    record = {
        field_name: 0.1 for field_name in FUNDAMENTAL_REQUIRED_FIELDS
    }

    readiness = _assess_symbol_records(
        branch="fundamental",
        symbols=["000001.SZ"],
        records={"000001.SZ": record},
        required_fields=FUNDAMENTAL_REQUIRED_FIELDS,
        manifest={
            "provider_status": "official_primary",
            "source_priority": "official_primary",
        },
        as_of="20240510",
    )

    assert readiness.status == STATUS_BLOCK
    assert readiness.source_priority == "official_primary"
    assert readiness.fallback_used is True
    assert readiness.blockers == ["fundamental_not_tushare_primary"]
