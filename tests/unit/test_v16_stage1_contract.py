from __future__ import annotations

from dataclasses import replace

import pytest

from quant_investor.v16.stage1_contract import PITFactRow, build_stage1_fact_package

SHA = "a" * 64


def _row(symbol: str, *, stratum: str = "large_liquid") -> PITFactRow:
    return PITFactRow(
        symbol=symbol,
        stratum=stratum,
        eligibility_receipt_sha256=SHA,
        formal_quant_score=0.2,
        quant_facts={"factor_snapshot_id": "q-1"},
        fundamental_facts={"statement_snapshot_id": "f-1"},
        macro_facts={"macro_generation_id": "m-1"},
    )


def test_stage1_fact_package_is_full_market_sorted_and_hash_bound() -> None:
    package = build_stage1_fact_package(
        rows=[_row("600000.SH", stratum="large"), _row("000001.SZ", stratum="mid")],
        funnel_symbols=["600000.SH"],
        cutoff_at="2026-07-17T07:00:00Z",
        expires_at="2026-07-18T07:00:00Z",
        pit_pointer_sha256=SHA,
    )
    assert [row.symbol for row in package.rows] == ["000001.SZ", "600000.SH"]
    assert package.funnel_symbols == ("600000.SH",)
    assert package.stratum_counts == {"large": 1, "mid": 1}
    package.verify()

    with pytest.raises(ValueError, match="payload SHA mismatch"):
        replace(package, funnel_symbols=("000001.SZ",)).verify()


def test_stage1_fact_package_rejects_unsealed_or_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="outside the eligible universe"):
        build_stage1_fact_package(
            rows=[_row("000001.SZ")],
            funnel_symbols=["600000.SH"],
            cutoff_at="2026-07-17T07:00:00Z",
            expires_at="2026-07-18T07:00:00Z",
            pit_pointer_sha256=SHA,
        )
    with pytest.raises(ValueError, match="later than"):
        build_stage1_fact_package(
            rows=[_row("000001.SZ")],
            funnel_symbols=["000001.SZ"],
            cutoff_at="2026-07-17T07:00:00Z",
            expires_at="2026-07-17T07:00:00Z",
            pit_pointer_sha256=SHA,
        )
    with pytest.raises(ValueError, match="non-finite"):
        PITFactRow(
            symbol="000001.SZ",
            stratum="large",
            eligibility_receipt_sha256=SHA,
            formal_quant_score=0.2,
            quant_facts={"bad": float("nan")},
            fundamental_facts={"ok": 1},
            macro_facts={"ok": 1},
        )
