from __future__ import annotations

import copy

import pandas as pd
import pytest

from quant_investor.market.rqdata_adapter import normalize_rqdata_daily_bars
from quant_investor.market.rqdata_shadow import (
    RQDataShadowContractError,
    build_rqdata_shadow_manifest,
    build_rqdata_shadow_request,
    measure_same_sample_reconciliation,
    validate_rqdata_shadow_manifest,
    validate_rqdata_shadow_request,
)


def _raw_frame() -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [
            ("000001.XSHE", pd.Timestamp("2026-08-27")),
            ("600000.XSHG", pd.Timestamp("2026-08-27")),
        ],
        names=["order_book_id", "date"],
    )
    return pd.DataFrame(
        {
            "open": [10.0, 20.0],
            "high": [11.0, 21.0],
            "low": [9.0, 19.0],
            "close": [10.5, 20.5],
            "volume": [1000.0, 2000.0],
            "total_turnover": [10500.0, 41000.0],
            "prev_close": [9.8, 19.8],
            "limit_up": [10.78, 21.78],
            "limit_down": [8.82, 17.82],
            "num_trades": [100.0, 200.0],
        },
        index=index,
    )


def _request() -> dict[str, object]:
    return build_rqdata_shadow_request(
        order_book_ids=["600000.XSHG", "000001.XSHE"],
        start_date="20260827",
        end_date="20260827",
    )


def test_shadow_request_is_deterministic_non_authorizing_and_exact() -> None:
    request = _request()

    assert request["order_book_ids"] == ["000001.XSHE", "600000.XSHG"]
    assert request["adjust_type"] == "none"
    assert request["provider_call_authorized"] is False
    assert request["canonical_write_authorized"] is False
    assert request["promotion_authorized"] is False
    assert len(str(request["request_sha256"])) == 64
    assert validate_rqdata_shadow_request(request) == request

    forged = {**request, "promotion_authorized": True}
    with pytest.raises(RQDataShadowContractError, match="replay exactly"):
        validate_rqdata_shadow_request(forged)


def test_shadow_manifest_binds_raw_canonical_request_and_provenance() -> None:
    raw = _raw_frame()
    canonical = normalize_rqdata_daily_bars(raw, adjustment_type="none")
    manifest = build_rqdata_shadow_manifest(
        request=_request(),
        raw_frame=raw,
        canonical_frame=canonical,
        acquired_at_utc="2026-08-30T08:00:00Z",
        provider_client_version="3.0.0",
        code_commit="a" * 40,
        run_id="rqdata-shadow-20260830-v1",
    )

    assert manifest["role"] == "primary_candidate"
    assert manifest["raw_row_count"] == 2
    assert manifest["canonical_row_count"] == 2
    assert manifest["requested_symbol_count"] == 2
    assert manifest["requested_symbol_coverage_ratio"] == 1.0
    assert manifest["missing_provider_symbols"] == []
    assert manifest["canonical_write_authorized"] is False
    assert manifest["promotion_authorized"] is False
    assert (
        validate_rqdata_shadow_manifest(
            manifest,
            request=_request(),
            raw_frame=raw,
            canonical_frame=canonical,
        )
        == manifest
    )

    changed = canonical.copy()
    changed.loc[0, "close"] = 99.0
    with pytest.raises(RQDataShadowContractError, match="replay exactly"):
        validate_rqdata_shadow_manifest(
            manifest,
            request=_request(),
            raw_frame=raw,
            canonical_frame=changed,
        )


def test_manifest_rejects_unrequested_or_out_of_range_rows() -> None:
    raw = _raw_frame()
    canonical = normalize_rqdata_daily_bars(raw, adjustment_type="none")
    canonical.loc[0, "provider_symbol"] = "000002.XSHE"
    with pytest.raises(RQDataShadowContractError, match="unrequested"):
        build_rqdata_shadow_manifest(
            request=_request(),
            raw_frame=raw,
            canonical_frame=canonical,
            acquired_at_utc="2026-08-30T08:00:00Z",
            provider_client_version="3.0.0",
            code_commit="a" * 40,
            run_id="rqdata-shadow-20260830-v1",
        )


def test_manifest_records_incomplete_requested_symbol_coverage() -> None:
    raw = _raw_frame().iloc[[0]].copy()
    canonical = normalize_rqdata_daily_bars(raw, adjustment_type="none")
    manifest = build_rqdata_shadow_manifest(
        request=_request(),
        raw_frame=raw,
        canonical_frame=canonical,
        acquired_at_utc="2026-08-30T08:00:00Z",
        provider_client_version="3.0.0",
        code_commit="a" * 40,
        run_id="rqdata-shadow-20260830-partial",
    )

    assert manifest["requested_symbol_coverage_ratio"] == 0.5
    assert manifest["missing_provider_symbols"] == ["600000.XSHG"]


def test_same_sample_reconciliation_measures_without_promotion_decision() -> None:
    rqdata = normalize_rqdata_daily_bars(_raw_frame(), adjustment_type="none")
    tushare = rqdata[["ts_code", "trade_date", "close", "vol", "amount"]].copy()
    tushare.loc[1, "close"] = 20.4
    result = measure_same_sample_reconciliation(
        rqdata_frame=rqdata,
        tushare_frame=tushare,
    )

    assert result["common_key_count"] == 2
    assert result["rqdata_only_key_count"] == 0
    assert result["tushare_only_key_count"] == 0
    assert result["measurements"]["close"]["exact_match_count"] == 1
    assert result["measurements"]["close"]["max_absolute_difference"] == pytest.approx(0.1)
    assert result["assessment_state"] == "MEASURED_NOT_EVALUATED"
    assert result["promotion_authorized"] is False


def test_reconciliation_rejects_duplicate_or_nonfinite_inputs() -> None:
    rqdata = normalize_rqdata_daily_bars(_raw_frame(), adjustment_type="none")
    tushare = rqdata[["ts_code", "trade_date", "close", "vol", "amount"]].copy()
    duplicate = pd.concat([tushare, tushare.iloc[[0]]], ignore_index=True)
    with pytest.raises(RQDataShadowContractError, match="duplicate"):
        measure_same_sample_reconciliation(rqdata_frame=rqdata, tushare_frame=duplicate)

    nonfinite = copy.deepcopy(tushare)
    nonfinite.loc[0, "amount"] = float("inf")
    with pytest.raises(RQDataShadowContractError, match="non-finite"):
        measure_same_sample_reconciliation(rqdata_frame=rqdata, tushare_frame=nonfinite)

    with pytest.raises(RQDataShadowContractError, match="empty"):
        measure_same_sample_reconciliation(
            rqdata_frame=rqdata.iloc[0:0],
            tushare_frame=tushare,
        )
