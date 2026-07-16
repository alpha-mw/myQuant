from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from quant_investor.market.cn_nontrading_evidence import (
    BAK_DAILY_NONTRADING_CLASSIFICATION,
    build_bak_daily_nontrading_evidence,
    canonical_json_sha256,
    evidence_cache_path,
    read_evidence_cache,
    symbol_set_sha256,
    validate_bak_daily_nontrading_evidence,
    write_evidence_cache,
)


def _zero_row(**overrides):
    row = {
        "ts_code": "000001.SZ",
        "trade_date": "20260707",
        "open": 0.0,
        "high": 0.0,
        "low": 0.0,
        "close": 12.34,
        "pre_close": 12.34,
        "change": 0.0,
        "pct_change": 0.0,
        "vol": 0.0,
        "amount": 0.0,
    }
    row.update(overrides)
    return row


def _build(frame: pd.DataFrame, *, pit_membership_sha256: str = "a" * 64):
    return build_bak_daily_nontrading_evidence(
        frame,
        trade_date="20260707",
        primary_missing_symbols=["000001.SZ"],
        query_params={"trade_date": "20260707"},
        pit_membership_path="data/parquet/cn/reference/stock_basic_membership.parquet",
        pit_membership_sha256=pit_membership_sha256,
    )


def test_exact_zero_bak_daily_row_is_typed_nontrading_evidence():
    payload = _build(pd.DataFrame([_zero_row()]))

    assert payload["classification"] == BAK_DAILY_NONTRADING_CLASSIFICATION
    assert payload["verified_symbols"] == ["000001.SZ"]
    assert payload["rejected_symbols"] == {}
    assert payload["writes_synthetic_bars"] is False
    assert payload["regulatory_suspension_claimed"] is False
    assert validate_bak_daily_nontrading_evidence(
        payload,
        trade_date="20260707",
        primary_missing_symbols=["000001.SZ"],
        pit_membership_sha256="a" * 64,
    ) == []


def test_nonzero_or_wrong_date_bak_daily_rows_are_rejected():
    nonzero = _build(pd.DataFrame([_zero_row(vol=1.0)]))
    wrong_date = _build(
        pd.DataFrame([_zero_row(trade_date="20260706")])
    )

    assert nonzero["verified_symbols"] == []
    assert "vol_nonzero_or_invalid" in nonzero["rejected_symbols"]["000001.SZ"]
    assert wrong_date["verified_symbols"] == []
    assert "trade_date_mismatch" in wrong_date["rejected_symbols"]["000001.SZ"]


def test_duplicate_and_close_mismatch_rows_are_rejected():
    duplicate = _build(pd.DataFrame([_zero_row(), _zero_row()]))
    close_mismatch = _build(pd.DataFrame([_zero_row(close=12.35)]))

    assert duplicate["rejected_symbols"]["000001.SZ"] == [
        "duplicate_exact_rows"
    ]
    assert "close_pre_close_mismatch" in close_mismatch["rejected_symbols"][
        "000001.SZ"
    ]


def test_hash_bound_cache_rejects_tampering(tmp_path):
    path = tmp_path / "evidence.json"
    payload = _build(pd.DataFrame([_zero_row()]))
    write_evidence_cache(path, payload)

    cached, blockers = read_evidence_cache(
        path,
        trade_date="20260707",
        primary_missing_symbols=["000001.SZ"],
        pit_membership_sha256="a" * 64,
    )
    assert blockers == []
    assert cached["verified_symbols"] == ["000001.SZ"]

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["verified_symbols"] = []
    path.write_text(json.dumps(tampered), encoding="utf-8")
    _cached, blockers = read_evidence_cache(
        path,
        trade_date="20260707",
        primary_missing_symbols=["000001.SZ"],
        pit_membership_sha256="a" * 64,
    )
    assert "payload_sha256_mismatch" in blockers
    assert "verified_symbols_sha256_mismatch" in blockers


def test_bak_cache_isolated_by_full_pit_sha_without_overwriting_old_bytes(
    tmp_path: Path,
) -> None:
    symbols = ["000001.SZ"]
    path_kwargs = {
        "trade_date": "20260707",
        "primary_missing_symbols": symbols,
    }
    legacy_path = evidence_cache_path(tmp_path, **path_kwargs)
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_bytes = b"legacy-unversioned-cache\n"
    legacy_path.write_bytes(legacy_bytes)

    first_path = evidence_cache_path(
        tmp_path,
        **path_kwargs,
        pit_membership_sha256="a" * 64,
    )
    write_evidence_cache(first_path, _build(pd.DataFrame([_zero_row()])))
    first_bytes = first_path.read_bytes()

    second_path = evidence_cache_path(
        tmp_path,
        **path_kwargs,
        pit_membership_sha256="b" * 64,
    )
    write_evidence_cache(
        second_path,
        _build(
            pd.DataFrame([_zero_row()]),
            pit_membership_sha256="b" * 64,
        ),
    )

    assert first_path != second_path
    assert f"pit_{'a' * 64}" in first_path.parts
    assert f"pit_{'b' * 64}" in second_path.parts
    assert legacy_path.read_bytes() == legacy_bytes
    assert first_path.read_bytes() == first_bytes


def test_bak_cache_path_rejects_incomplete_explicit_pit_sha(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="complete 64-character SHA-256"):
        evidence_cache_path(
            tmp_path,
            trade_date="20260707",
            primary_missing_symbols=["000001.SZ"],
            pit_membership_sha256="a" * 63,
        )


def test_recomputed_hash_cannot_turn_an_unrelated_symbol_into_evidence():
    payload = _build(pd.DataFrame([_zero_row()]))
    payload["verified_symbols"] = ["999999.SZ"]
    payload["verified_symbols_sha256"] = symbol_set_sha256(["999999.SZ"])
    payload["matched_records"][0]["ts_code"] = "999999.SZ"
    payload["matched_records_sha256"] = canonical_json_sha256(
        payload["matched_records"]
    )
    unsigned = dict(payload)
    unsigned.pop("payload_sha256")
    payload["payload_sha256"] = canonical_json_sha256(unsigned)

    blockers = validate_bak_daily_nontrading_evidence(
        payload,
        trade_date="20260707",
        primary_missing_symbols=["000001.SZ"],
        pit_membership_sha256="a" * 64,
    )

    assert "verified_symbols_outside_primary_missing" in blockers
    assert "candidate_classification_union_mismatch" in blockers
