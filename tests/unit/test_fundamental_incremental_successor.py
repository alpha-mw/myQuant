from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_investor.market import fundamental_incremental as incremental_module
from quant_investor.market.fundamental_incremental import (
    RAW_TABLES,
    SUCCESSOR_APPEND_FIRST_MODE,
    SUCCESSOR_PROVENANCE_SCHEMA,
    SafeSuccessorError,
    assemble_safe_successor,
    build_keyset_closure,
    build_successor_chain,
    seal_successor_provider_manifest,
    seal_support_plan,
    stage_successor_generation,
    validate_successor_provenance,
)
from quant_investor.market.fundamental_provider_contract import (
    canonical_json_sha256,
    frame_fingerprint,
)


SYMBOLS = ("000001.SZ", "000002.SZ", "000003.SZ")
PARENT_CUTOFF = "20260806"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _support_bytes() -> dict[str, bytes]:
    return {
        "predecessor_pointer": (
            b'{"generation_id":"parent_v2","schema_version":'
            b'"cn-fundamental-pointer.v1","status":"OK"}\n'
        ),
        "predecessor_manifest": (
            b'{"generation_id":"parent_v2","schema_version":'
            b'"cn-fundamental-generation.v1","status":"OK"}\n'
        ),
        "support_manifest": (
            b'{"schema_version":"fixture-successor-support.v1",'
            b'"status":"sealed"}\n'
        ),
    }


def _support_refs() -> dict[str, dict[str, str]]:
    return {
        name: {"path": f"sealed/{name}.json", "sha256": _sha(payload)}
        for name, payload in _support_bytes().items()
    }


def _period_values() -> dict[str, float]:
    return {
        "fin_roe": 0.10,
        "fin_roa": 0.05,
        "fin_debt_to_assets": 0.40,
        "fin_net_profit_yoy": 0.20,
        "fin_ocf_to_profit": 0.50,
        "fin_fcf_to_profit": 0.40,
        "free_cashflow": 40.0,
    }


def _parent_tables() -> dict[str, pd.DataFrame]:
    values = _period_values()
    period = pd.DataFrame(
        [
            {
                "ts_code": symbol,
                "end_date": "20251231",
                "availability_date": pd.Timestamp("2026-04-01"),
                "source_version": "2026-04-01",
                "source": "fixture",
                "fetched_at": "2026-04-01T08:00:00Z",
                **values,
            }
            for symbol in SYMBOLS
        ]
    )
    rows = []
    for position, symbol in enumerate(SYMBOLS, start=1):
        total_mv = float(position * 1_000_000)
        rows.append(
            {
                "ts_code": symbol,
                "trade_date": pd.Timestamp("2026-08-06"),
                "end_date": "20251231",
                "availability_date": pd.Timestamp("2026-04-01"),
                "source_version": "2026-04-01",
                "source": "fixture",
                "fetched_at": "2026-04-01T08:00:00Z",
                "sector": "fixture-sector",
                "size_bucket": ("small", "large", "large")[position - 1],
                "total_mv_rmb": total_mv,
                **values,
                "fcf_to_price": 40.0 / total_mv,
                "forecast_revision": 0.10,
                "forecast_end_date": "20251231",
                "forecast_ann_date": "2026-03-01",
                "forecast_type": "预增",
                "forecast_summary": "base",
                "forecast_change_reason": "base",
                "forecast_source": "fixture",
                "forecast_fetched_at": "2026-03-01T08:00:00Z",
                "forecast_ingest_run_id": "parent",
            }
        )
    daily = pd.DataFrame(rows)
    return {
        "fundamental_period": period,
        "fundamental_daily": daily,
        "fundamental_quarantine": pd.DataFrame(),
    }


def _parent_closure(
    tables: dict[str, pd.DataFrame],
    *,
    generation_id: str = "parent_v2",
    cutoff: str = PARENT_CUTOFF,
    primary_provenance: dict | None = None,
) -> dict:
    return {
        "generation_id": generation_id,
        "cutoff": cutoff,
        "pointer_sha256": _sha(_support_bytes()["predecessor_pointer"]),
        "manifest_sha256": _sha(_support_bytes()["predecessor_manifest"]),
        "table_sha256": {
            "fundamental_period": "c" * 64,
            "fundamental_daily": "d" * 64,
            "fundamental_quarantine": "e" * 64,
        },
        "table_frame_fingerprints": {
            name: frame_fingerprint(frame) for name, frame in tables.items()
        },
        "primary_provenance": primary_provenance
        or {"schema_version": "cn-fundamental-primary-provenance.v2"},
    }


def _financial_row(
    table: str,
    symbol: str,
    end_date: str,
    ann_date: str,
    *,
    delta: bool,
) -> dict:
    common = {
        "ts_code": symbol,
        "end_date": end_date,
        "ann_date": ann_date,
        "source": "fixture",
        "fetched_at": f"{ann_date}T08:00:00Z",
    }
    if table == "fina_indicator":
        return {
            **common,
            "roe_dt": 12.0 if delta else 10.0,
            "roe": 12.0 if delta else 10.0,
            "roa": 6.0 if delta else 5.0,
            "debt_to_assets": 45.0 if delta else 40.0,
            "netprofit_yoy": 22.0 if delta else 20.0,
        }
    if table == "income":
        return {**common, "n_income_attr_p": 120.0 if delta else 100.0}
    if table == "balancesheet":
        return {
            **common,
            "total_liab": 45.0 if delta else 40.0,
            "total_assets": 100.0,
        }
    if table == "cashflow":
        return {
            **common,
            "n_cashflow_act": 60.0 if delta else 50.0,
            "c_pay_acq_const_fiolta": 10.0,
            "free_cashflow": np.nan,
        }
    raise AssertionError(table)


def _raw_tables(target: str = "20260808") -> dict[str, pd.DataFrame]:
    raw: dict[str, pd.DataFrame] = {name: pd.DataFrame() for name in RAW_TABLES}
    for table in RAW_TABLES[:4]:
        records = []
        for symbol in SYMBOLS:
            records.append(
                _financial_row(
                    table,
                    symbol,
                    "20251231",
                    "20260401",
                    delta=False,
                )
            )
            records.append(
                _financial_row(
                    table,
                    symbol,
                    "20260630",
                    "20260807",
                    delta=True,
                )
            )
            if table == "income":
                records.append(
                    {
                        "ts_code": symbol,
                        "end_date": "20250630",
                        "ann_date": "20250701",
                        "n_income_attr_p": 100.0,
                        "source": "fixture",
                        "fetched_at": "2025-07-01T08:00:00Z",
                    }
                )
        raw[table] = pd.DataFrame(records)
    trade_dates = ["20260807"] + (["20260808"] if target >= "20260808" else [])
    raw["daily_basic"] = pd.DataFrame(
        [
            {
                "ts_code": symbol,
                "trade_date": trade_date,
                "total_mv": DecimalText,
                "sector": "fixture-sector",
            }
            for trade_date in trade_dates
            for symbol, DecimalText in zip(SYMBOLS, ("100", "200", "300"))
        ]
    )
    forecast = []
    for symbol in SYMBOLS:
        forecast.append(
            {
                "ts_code": symbol,
                "end_date": "20251231",
                "ann_date": "20260301",
                "p_change_min": 10.0,
                "p_change_max": 10.0,
                "type": "预增",
                "summary": "base",
                "change_reason": "base",
                "source": "fixture",
                "fetched_at": "2026-03-01T08:00:00Z",
            }
        )
    forecast.extend(
        [
            {
                "ts_code": "000001.SZ",
                "end_date": "20260630",
                "ann_date": "20260807",
                "p_change_min": 30.0,
                "p_change_max": 30.0,
                "type": "预增",
                "summary": "first",
                "change_reason": "first",
                "source": "fixture",
                "fetched_at": "2026-08-07T08:00:00Z",
            },
            {
                "ts_code": "000001.SZ",
                "end_date": "20260630",
                "ann_date": "20260808",
                "p_change_min": 50.0,
                "p_change_max": 50.0,
                "type": "预增",
                "summary": "second",
                "change_reason": "second",
                "source": "fixture",
                "fetched_at": "2026-08-08T08:00:00Z",
            },
        ]
    )
    raw["forecast"] = pd.DataFrame(
        [row for row in forecast if row["ann_date"] <= target]
    )
    return raw


def _keyset(raw: dict[str, pd.DataFrame], *, true_missing=()) -> dict:
    keys = [
        (row.ts_code, row.trade_date)
        for row in raw["daily_basic"].itertuples(index=False)
        if row.trade_date > PARENT_CUTOFF
    ]
    return build_keyset_closure(
        observed_bar_keys=keys,
        daily_basic_keys=keys,
        true_missing_keys=true_missing,
    )


def _bundle(
    *,
    target: str = "20260808",
    parents: dict[str, pd.DataFrame] | None = None,
    raw: dict[str, pd.DataFrame] | None = None,
    closure: dict | None = None,
    plan_extra: dict | None = None,
    keyset: dict | None = None,
):
    parents = parents or _parent_tables()
    raw = raw or _raw_tables(target)
    closure = closure or _parent_closure(parents)
    plan = seal_support_plan(
        raw,
        parent_cutoff=PARENT_CUTOFF,
        target_cutoff=target,
        permanent_support_refs=_support_refs(),
        extra=plan_extra,
    )
    return assemble_safe_successor(
        parent_tables=parents,
        parent_closure=closure,
        support_raw_tables=raw,
        plan_metadata=plan,
        keyset_closure=keyset or _keyset(raw),
        parent_cutoff=PARENT_CUTOFF,
        target_cutoff=target,
        run_id=f"successor_{target}",
    )


class _NoAggregateAccessStore(Mapping[str, object]):
    def __init__(self, tables: Mapping[str, pd.DataFrame], batch_rows: int = 2) -> None:
        self._tables = {name: frame.copy() for name, frame in tables.items()}
        self._batch_rows = batch_rows
        self.getitem_calls = 0
        self.table_fingerprints = {
            name: frame_fingerprint(frame) for name, frame in self._tables.items()
        }

    def __iter__(self) -> Iterator[str]:
        return iter(RAW_TABLES)

    def __len__(self) -> int:
        return len(RAW_TABLES)

    def __getitem__(self, key: str) -> object:
        self.getitem_calls += 1
        raise AssertionError(f"aggregate access forbidden: {key}")

    def iter_batches(
        self,
        table: str,
        *,
        batch_rows: int,
        batch_bytes: int,
    ) -> Iterator[pd.DataFrame]:
        assert batch_rows >= self._batch_rows
        assert batch_bytes > 0
        frame = self._tables[table]
        for start in range(0, len(frame), self._batch_rows):
            yield frame.iloc[start : start + self._batch_rows].copy()


def _derivation_budget(raw: Mapping[str, pd.DataFrame]) -> dict:
    body = {
        "schema_version": (
            "cn-fundamental-successor-derivation-accumulator-budget.v1"
        ),
        "status": "PASS",
        "period_anchor_row_limit": len(SYMBOLS),
        "forecast_anchor_row_limit": len(SYMBOLS),
        "period_delta_row_limit": sum(len(raw[table]) for table in RAW_TABLES[:4]),
        "period_lineage_row_limit": sum(len(raw[table]) for table in RAW_TABLES[:4]),
        "forecast_delta_row_limit": len(raw["forecast"]),
        "daily_basic_row_limit": len(raw["daily_basic"]),
        "total_accumulator_byte_limit": 64 * 1024 * 1024,
        "effective_memory_headroom_bytes": 1024 * 1024 * 1024,
        "post_capture_receipt_sha256": "a" * 64,
    }
    body["binding_sha256"] = canonical_json_sha256(body)
    return body


def _sorted_raw(raw: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {
        table: frame.sort_values(
            [
                column
                for column in ("ts_code", "ann_date", "trade_date", "end_date")
                if column in frame.columns
            ],
            kind="mergesort",
        ).reset_index(drop=True)
        if not frame.empty
        else frame.copy()
        for table, frame in raw.items()
    }


def _streamed_bundle(
    raw: Mapping[str, pd.DataFrame],
    *,
    budget: Mapping[str, object] | None = None,
):
    sorted_raw = _sorted_raw(raw)
    store = _NoAggregateAccessStore(sorted_raw)
    plan = seal_support_plan(
        store,
        parent_cutoff=PARENT_CUTOFF,
        target_cutoff="20260808",
        permanent_support_refs=_support_refs(),
        extra={
            "derivation_resource_budget": dict(
                budget or _derivation_budget(sorted_raw)
            )
        },
    )
    bundle = assemble_safe_successor(
        parent_tables=_parent_tables(),
        parent_closure=_parent_closure(_parent_tables()),
        support_raw_tables=store,
        plan_metadata=plan,
        keyset_closure=_keyset(sorted_raw),
        parent_cutoff=PARENT_CUTOFF,
        target_cutoff="20260808",
        run_id="successor_20260808",
    )
    return bundle, store, sorted_raw


def test_streamed_support_is_reference_equivalent_without_aggregate_getitem() -> None:
    sorted_raw = _sorted_raw(_raw_tables("20260808"))
    reference = _bundle(target="20260808", raw=sorted_raw)
    streamed, store, _raw = _streamed_bundle(sorted_raw)

    assert store.getitem_calls == 0
    assert frame_fingerprint(streamed.period_suffix) == frame_fingerprint(
        reference.period_suffix
    )
    assert frame_fingerprint(streamed.daily_suffix) == frame_fingerprint(
        reference.daily_suffix
    )
    assert streamed.lineage["period_rows"] == reference.lineage["period_rows"]
    streaming = streamed.lineage["support_streaming"]
    assert all(
        streaming[lane]["full_table_getitem_used"] is False
        for lane in ("financial", "forecast", "daily_basic")
    )
    assert streaming["financial"]["maximum_resident_source_rows"] <= 9


def test_deep_prefix_retains_only_boundary_anchors_and_delta() -> None:
    shallow = _sorted_raw(_raw_tables("20260808"))
    deep = {table: frame.copy() for table, frame in shallow.items()}
    quarter_ends = ("0331", "0630", "0930", "1231")
    for table in RAW_TABLES[:4]:
        historical = []
        for symbol in SYMBOLS:
            for year in range(2010, 2025):
                for quarter_end in quarter_ends:
                    historical.append(
                        _financial_row(
                            table,
                            symbol,
                            f"{year}{quarter_end}",
                            f"{year + 1}0101",
                            delta=False,
                        )
                    )
        deep[table] = pd.concat(
            [deep[table], pd.DataFrame(historical)], ignore_index=True
        )
    historical_forecast = []
    for symbol in SYMBOLS:
        for year in range(2010, 2025):
            for quarter_end in quarter_ends:
                historical_forecast.append(
                    {
                        "ts_code": symbol,
                        "end_date": f"{year}{quarter_end}",
                        "ann_date": f"{year + 1}0101",
                        "p_change_min": 10.0,
                        "p_change_max": 10.0,
                        "type": "预增",
                        "summary": "historical",
                        "change_reason": "historical",
                        "source": "fixture",
                        "fetched_at": f"{year + 1}-01-01T08:00:00Z",
                    }
                )
    deep["forecast"] = pd.concat(
        [deep["forecast"], pd.DataFrame(historical_forecast)], ignore_index=True
    )
    deep = _sorted_raw(deep)

    shallow_bundle, _shallow_store, _shallow_raw = _streamed_bundle(shallow)
    deep_reference = _bundle(target="20260808", raw=deep)
    deep_bundle, deep_store, _deep_raw = _streamed_bundle(deep)

    assert deep_store.getitem_calls == 0
    assert frame_fingerprint(deep_bundle.period_suffix) == frame_fingerprint(
        deep_reference.period_suffix
    )
    assert frame_fingerprint(deep_bundle.daily_suffix) == frame_fingerprint(
        deep_reference.daily_suffix
    )
    assert frame_fingerprint(deep_bundle.period_suffix) == frame_fingerprint(
        shallow_bundle.period_suffix
    )
    assert frame_fingerprint(deep_bundle.daily_suffix) == frame_fingerprint(
        shallow_bundle.daily_suffix
    )
    shallow_streaming = shallow_bundle.lineage["support_streaming"]
    deep_streaming = deep_bundle.lineage["support_streaming"]
    expected_counts = {
        "daily_basic": 6,
        "forecast_anchor": 3,
        "forecast_delta": 2,
        "period_anchor": 3,
        "period_delta": 3,
        "period_lineage": 3,
    }
    assert shallow_streaming["accumulator"]["counts"] == expected_counts
    assert deep_streaming["accumulator"]["counts"] == expected_counts
    assert deep_streaming["financial"]["retained_prefix_winners"] == len(SYMBOLS)
    assert deep_streaming["financial"]["retained_delta_records"] == 3
    assert deep_streaming["financial"]["retained_delta_lineage"] == 3
    assert deep_streaming["forecast"]["retained_prefix_winners"] == len(SYMBOLS)
    assert deep_streaming["forecast"]["retained_delta_records"] == 2
    assert deep_streaming["financial"]["full_support_records_retained"] is False
    assert deep_streaming["forecast"]["full_support_records_retained"] is False
    assert (
        deep_streaming["financial"]["maximum_resident_source_rows"]
        > shallow_streaming["financial"]["maximum_resident_source_rows"]
    )


def test_streamed_derivation_fails_closed_at_sealed_accumulator_limit() -> None:
    raw = _sorted_raw(_raw_tables("20260808"))
    budget = _derivation_budget(raw)
    budget["daily_basic_row_limit"] = len(raw["daily_basic"]) - 1
    budget.pop("binding_sha256")
    budget["binding_sha256"] = canonical_json_sha256(budget)

    with pytest.raises(SafeSuccessorError) as failure:
        _streamed_bundle(raw, budget=budget)
    assert failure.value.code == "DERIVATION_ACCUMULATOR_LIMIT_EXCEEDED"


def test_financial_symbol_cap_applies_across_all_four_support_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _sorted_raw(_raw_tables("20260808"))
    for table in RAW_TABLES[:4]:
        raw[table]["resource_padding"] = ""
        first = raw[table].index[raw[table]["ts_code"] == SYMBOLS[0]][0]
        raw[table].loc[first, "resource_padding"] = "x" * 9_000
    monkeypatch.setattr(incremental_module, "SUPPORT_STREAM_BATCH_BYTES", 16_384)
    monkeypatch.setattr(incremental_module, "SUPPORT_SYMBOL_MAX_BYTES", 32_768)

    with pytest.raises(SafeSuccessorError) as failure:
        _streamed_bundle(raw)
    assert failure.value.code == "SUPPORT_SYMBOL_RESOURCE_LIMIT"


def test_v2_to_v3_to_v3_flattens_chain_and_retains_original_seam() -> None:
    parents = _parent_tables()
    v2 = _parent_closure(parents)
    first = build_successor_chain(
        v2,
        parent_cutoff=PARENT_CUTOFF,
        target_cutoff="20260807",
        generation_id="successor_1",
    )
    v3 = _parent_closure(
        parents,
        generation_id="successor_1",
        cutoff="20260807",
        primary_provenance={
            "schema_version": SUCCESSOR_PROVENANCE_SCHEMA,
            "successor_chain": first,
        },
    )
    second = build_successor_chain(
        v3,
        parent_cutoff="20260807",
        target_cutoff="20260808",
        generation_id="successor_2",
    )
    assert second["original_seam"] == first["original_seam"]
    assert second["root_reference"] == first["root_reference"]
    assert second["immediate_predecessor"]["generation_id"] == "successor_1"
    assert second["append_boundary"]["parent_cutoff"] == "20260807"
    assert second["ancestor_generation_ids"] == [
        "parent_v2",
        "successor_1",
        "successor_2",
    ]


def test_boundary_anchor_mismatch_blocks_and_declared_nonreachability_passes() -> None:
    raw = _raw_tables("20260807")
    raw["fina_indicator"].loc[
        (raw["fina_indicator"]["ts_code"] == "000001.SZ")
        & (raw["fina_indicator"]["ann_date"] == "20260401"),
        "roe_dt",
    ] = 99.0
    with pytest.raises(SafeSuccessorError, match="BOUNDARY_ANCHOR_MISMATCH"):
        _bundle(target="20260807", raw=raw)

    raw = _raw_tables("20260807")
    for table in RAW_TABLES[:4]:
        raw[table] = raw[table][
            ~(
                (raw[table]["ts_code"] == "000001.SZ")
                & (raw[table]["ann_date"] <= PARENT_CUTOFF)
            )
        ].reset_index(drop=True)
    raw["forecast"] = raw["forecast"][
        ~(
            (raw["forecast"]["ts_code"] == "000001.SZ")
            & (raw["forecast"]["ann_date"] <= PARENT_CUTOFF)
        )
    ].reset_index(drop=True)
    plan = seal_support_plan(
        raw,
        parent_cutoff=PARENT_CUTOFF,
        target_cutoff="20260807",
        permanent_support_refs=_support_refs(),
        boundary_non_reachability={
            "period": ["000001.SZ"],
            "forecast": ["000001.SZ"],
        },
    )
    result = assemble_safe_successor(
        parent_tables=_parent_tables(),
        parent_closure=_parent_closure(_parent_tables()),
        support_raw_tables=raw,
        plan_metadata=plan,
        keyset_closure=_keyset(raw),
        parent_cutoff=PARENT_CUTOFF,
        target_cutoff="20260807",
        run_id="nonreachable",
    )
    assert result.lineage["boundary"]["period"]["000001.SZ"] == (
        "LANE_NON_REACHABLE_DECLARED"
    )


def test_hidden_previous_year_dependency_must_be_present_or_proven_absent() -> None:
    raw = _raw_tables("20260807")
    raw["fina_indicator"].loc[
        (raw["fina_indicator"]["ts_code"] == "000001.SZ")
        & (raw["fina_indicator"]["ann_date"] == "20260807"),
        "netprofit_yoy",
    ] = np.nan
    raw["income"] = raw["income"][
        ~(
            (raw["income"]["ts_code"] == "000001.SZ")
            & (raw["income"]["end_date"] == "20250630")
        )
    ].reset_index(drop=True)
    with pytest.raises(SafeSuccessorError, match="HIDDEN_DEPENDENCY_UNPROVEN"):
        _bundle(target="20260807", raw=raw)


def test_hidden_dependency_before_support_start_has_specific_stop_code() -> None:
    raw = _raw_tables("20260807")
    for table in ("fina_indicator", "income", "balancesheet", "cashflow"):
        selector = (
            (raw[table]["ts_code"] == "000001.SZ")
            & (raw[table]["ann_date"] == "20260807")
        )
        raw[table].loc[selector, "end_date"] = "20200101"
    raw["fina_indicator"].loc[
        (raw["fina_indicator"]["ts_code"] == "000001.SZ")
        & (raw["fina_indicator"]["ann_date"] == "20260807"),
        "netprofit_yoy",
    ] = np.nan
    with pytest.raises(SafeSuccessorError, match="SUPPORT_START_ANCHOR_UNCLOSED"):
        _bundle(
            target="20260807",
            raw=raw,
            plan_extra={"support_start": "20190806"},
        )


def test_previous_year_fallback_is_availability_aware() -> None:
    raw = _raw_tables("20260807")
    raw["fina_indicator"].loc[
        raw["fina_indicator"]["ann_date"] == "20260807", "netprofit_yoy"
    ] = np.nan
    bundle = _bundle(target="20260807", raw=raw)
    assert set(bundle.period_suffix["fin_net_profit_yoy"].round(12)) == {0.20}


def test_material_event_tie_and_invalid_cap_block() -> None:
    raw = _raw_tables("20260807")
    conflict = raw["income"].iloc[[0]].copy()
    conflict["n_income_attr_p"] = 999.0
    raw["income"] = pd.concat([raw["income"], conflict], ignore_index=True)
    with pytest.raises(SafeSuccessorError, match="MATERIAL_EVENT_TIE"):
        _bundle(target="20260807", raw=raw)

    raw = _raw_tables("20260807")
    raw["daily_basic"].loc[0, "total_mv"] = "0"
    with pytest.raises(SafeSuccessorError, match="INVALID_TOTAL_MV"):
        _bundle(target="20260807", raw=raw)


def test_same_day_batch_is_permutation_invariant() -> None:
    raw = _raw_tables("20260808")
    first = _bundle(raw=raw)
    permuted = {
        table: frame.sample(frac=1.0, random_state=42).reset_index(drop=True)
        for table, frame in raw.items()
    }
    second = _bundle(raw=permuted)
    assert frame_fingerprint(first.period_suffix) == frame_fingerprint(second.period_suffix)
    assert frame_fingerprint(first.daily_suffix) == frame_fingerprint(second.daily_suffix)


def test_future_revision_does_not_look_back_and_forecast_carries_then_supersedes() -> None:
    bundle = _bundle(target="20260808")
    symbol = bundle.daily_suffix[bundle.daily_suffix["ts_code"] == "000001.SZ"]
    by_date = {
        row.trade_date.strftime("%Y%m%d"): row
        for row in symbol.itertuples(index=False)
    }
    assert by_date["20260807"].forecast_revision == pytest.approx(0.30)
    assert by_date["20260807"].forecast_summary == "first"
    assert by_date["20260808"].forecast_revision == pytest.approx(0.50)
    assert by_date["20260808"].forecast_summary == "second"
    other = bundle.daily_suffix[bundle.daily_suffix["ts_code"] == "000002.SZ"]
    assert set(other["forecast_revision"]) == {0.10}


def test_decimal_size_ties_and_future_session_invariance() -> None:
    raw_one = _raw_tables("20260807")
    raw_one["daily_basic"]["total_mv"] = ["100", "100", "300"]
    one = _bundle(target="20260807", raw=raw_one)
    first_buckets = dict(zip(one.daily_suffix["ts_code"], one.daily_suffix["size_bucket"]))
    assert first_buckets == {
        "000001.SZ": "mid",
        "000002.SZ": "mid",
        "000003.SZ": "large",
    }

    raw_two = _raw_tables("20260808")
    raw_two["daily_basic"].loc[
        raw_two["daily_basic"]["trade_date"] == "20260807", "total_mv"
    ] = ["100", "100", "300"]
    two = _bundle(target="20260808", raw=raw_two)
    prior = two.daily_suffix[two.daily_suffix["trade_date"] == pd.Timestamp("2026-08-07")]
    assert dict(zip(prior["ts_code"], prior["size_bucket"])) == first_buckets


def test_parent_prefix_tamper_blocks_before_derivation() -> None:
    parents = _parent_tables()
    closure = _parent_closure(parents)
    parents["fundamental_daily"].loc[0, "fin_roe"] = -0.0
    with pytest.raises(SafeSuccessorError, match="PARENT_PREFIX_TAMPER"):
        _bundle(target="20260807", parents=parents, closure=closure)


def test_no_period_state_is_counted_but_true_missing_blocks() -> None:
    raw = _raw_tables("20260807")
    raw["daily_basic"] = pd.concat(
        [
            raw["daily_basic"],
            pd.DataFrame(
                [
                    {
                        "ts_code": "000004.SZ",
                        "trade_date": "20260807",
                        "total_mv": "400",
                        "sector": "fixture-sector",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    bundle = _bundle(target="20260807", raw=raw)
    assert bundle.readiness["NO_PERIOD_STATE"] == 1
    assert bundle.keyset_closure["no_period_state_keys"] == ["000004.SZ|20260807"]
    assert len(bundle.daily_suffix) == 3

    missing_key = [("000005.SZ", "20260807")]
    with pytest.raises(SafeSuccessorError, match="TRUE_MISSING_NOT_ZERO"):
        _bundle(
            target="20260807",
            raw=_raw_tables("20260807"),
            keyset=_keyset(_raw_tables("20260807"), true_missing=missing_key),
        )


def test_nonbar_scope_symbols_are_included_in_boundary_proof() -> None:
    raw = _raw_tables("20260807")
    observed = [
        (row.ts_code, row.trade_date)
        for row in raw["daily_basic"].itertuples(index=False)
    ]
    keyset = build_keyset_closure(
        observed_bar_keys=observed,
        daily_basic_keys=observed,
        suspended_keys=[("000004.SZ", "20260807")],
    )
    bundle = _bundle(target="20260807", raw=raw, keyset=keyset)
    assert bundle.lineage["boundary"]["period"]["000004.SZ"] == (
        "LANE_NON_REACHABLE_EMPTY"
    )
    assert bundle.lineage["boundary"]["forecast"]["000004.SZ"] == (
        "LANE_NON_REACHABLE_EMPTY"
    )


def test_successor_chain_rejects_cycle_and_self_reference() -> None:
    parents = _parent_tables()
    v2 = _parent_closure(parents)
    with pytest.raises(SafeSuccessorError, match="SUCCESSOR_SELF_REFERENCE"):
        build_successor_chain(
            v2,
            parent_cutoff=PARENT_CUTOFF,
            target_cutoff="20260807",
            generation_id="parent_v2",
        )
    first = build_successor_chain(
        v2,
        parent_cutoff=PARENT_CUTOFF,
        target_cutoff="20260807",
        generation_id="successor_1",
    )
    first["ancestor_generation_ids"].append("parent_v2")
    body = dict(first)
    body.pop("chain_fingerprint")
    first["chain_fingerprint"] = canonical_json_sha256(body)
    v3 = _parent_closure(
        parents,
        generation_id="successor_1",
        cutoff="20260807",
        primary_provenance={
            "schema_version": SUCCESSOR_PROVENANCE_SCHEMA,
            "successor_chain": first,
        },
    )
    with pytest.raises(SafeSuccessorError, match="SUCCESSOR_CHAIN_CYCLE"):
        build_successor_chain(
            v3,
            parent_cutoff="20260807",
            target_cutoff="20260808",
            generation_id="successor_2",
        )


def _path_backed_case(tmp_path: Path):
    parents = _parent_tables()
    parent_root = tmp_path / "parent"
    parent_root.mkdir()
    parent_paths = {}
    for name, frame in parents.items():
        path = parent_root / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        parent_paths[name] = path
    closure = _parent_closure(parents)
    closure["table_sha256"] = {
        name: _sha(path.read_bytes()) for name, path in parent_paths.items()
    }
    closure["validated_frame_fingerprints"] = dict(
        closure["table_frame_fingerprints"]
    )
    raw = _raw_tables("20260807")
    plan = seal_support_plan(
        raw,
        parent_cutoff=PARENT_CUTOFF,
        target_cutoff="20260807",
        permanent_support_refs=_support_refs(),
    )
    bundle = assemble_safe_successor(
        parent_tables=parent_paths,
        parent_closure=closure,
        support_raw_tables=raw,
        plan_metadata=plan,
        keyset_closure=_keyset(raw),
        parent_cutoff=PARENT_CUTOFF,
        target_cutoff="20260807",
        run_id="staged_successor",
        staging_parent=tmp_path,
    )
    market_manifest = tmp_path / "market_manifest.json"
    market_manifest.write_bytes(b'{"snapshot_id":"market-20260807"}\n')
    pit_membership = tmp_path / "pit_membership.parquet"
    pit_membership.write_bytes(b"immutable PIT membership fixture\n")
    expected_scope = tmp_path / "expected_scope.json"
    expected_scope.write_bytes(b'{"as_of":"20260807","true_missing":0}\n')
    market_pointer = tmp_path / "market_pointer.json"
    market_pointer.write_bytes(
        b'{"manifest_path":"immutable/market_manifest.json",'
        b'"snapshot_id":"market-20260807"}\n'
    )
    pit_pointer = tmp_path / "pit_pointer.json"
    pit_pointer.write_bytes(
        b'{"generation_id":"pit-20260807",'
        b'"membership_path":"immutable/pit_membership.parquet"}\n'
    )
    target_bindings = {
        "market_pointer": {
            "path": str(market_pointer),
            "sha256": _sha(market_pointer.read_bytes()),
            "as_of": "20260807",
            "immutable_refs": [
                {
                    "path": str(market_manifest),
                    "sha256": _sha(market_manifest.read_bytes()),
                }
            ],
        },
        "pit_pointer": {
            "path": str(pit_pointer),
            "sha256": _sha(pit_pointer.read_bytes()),
            "as_of": "20260807",
            "immutable_refs": [
                {
                    "path": str(pit_membership),
                    "sha256": _sha(pit_membership.read_bytes()),
                }
            ],
        },
        "pit_membership": {
            "path": str(pit_membership),
            "sha256": _sha(pit_membership.read_bytes()),
            "as_of": "20260807",
        },
        "expected_scope": {
            "path": str(expected_scope),
            "sha256": _sha(expected_scope.read_bytes()),
            "as_of": "20260807",
        },
    }
    support_files = {
        _support_refs()[name]["path"]: payload
        for name, payload in _support_bytes().items()
    }
    evidence_sha = {name: _sha(payload) for name, payload in support_files.items()}
    provider = seal_successor_provider_manifest(
        bundle,
        provider="fixture",
        request_receipts_sha256="f" * 64,
        evidence_files=evidence_sha,
    )
    return bundle, target_bindings, support_files, provider


def test_staging_exact_readback_and_permanent_support_tamper_block(tmp_path: Path) -> None:
    bundle, targets, support_files, provider = _path_backed_case(tmp_path)
    capture = stage_successor_generation(
        bundle,
        staging_root=tmp_path / "staging",
        generation_id="staged_successor",
        provider_manifest=provider,
        target_bindings=targets,
        provider_evidence_files=support_files,
    )
    pointer = json.loads(capture.pointer_bytes)
    manifest = json.loads(capture.manifest_bytes)
    validated = validate_successor_provenance(
        pointer,
        manifest,
        generation_root=capture.staging_root,
        historical_only=True,
    )
    assert validated["machine_states"] == {
        "mixed": True,
        "legacy_direct_reader_provenance": "limited",
        "binding_aware_research_ready": True,
        "homogeneous_history_ready": False,
    }
    readiness_state = {
        name: pointer["metadata"]["readiness"][name]
        for name in validated["machine_states"]
    }
    assert readiness_state == validated["machine_states"]
    assert pointer["metadata"]["readiness"]["gate2_contract"] == (
        "cn-fundamental-readiness.safe-successor.v1"
    )
    assert pointer["metadata"]["readiness"]["prefix_gate_passed"] is True
    assert pointer["metadata"]["readiness"]["suffix_gate_passed"] is True
    assert pointer["metadata"]["readiness"]["structural_gate_passed"] is True
    assert pointer["primary_provenance"]["mixed_generation"] is True
    assert pointer["primary_provenance"]["seam_trade_date"] == PARENT_CUTOFF
    assert pointer["primary_provenance"]["suffix_contract"] == (
        "cn-fundamental-derivation.safe-successor.v1"
    )
    for document in (pointer, manifest):
        metadata_state = {
            name: document["metadata"][name]
            for name in validated["machine_states"]
        }
        assert metadata_state == validated["machine_states"]
    altered_pointer = copy.deepcopy(pointer)
    altered_manifest = copy.deepcopy(manifest)
    for document in (altered_pointer, altered_manifest):
        envelope = document["primary_provenance"]
        envelope["machine_states"]["binding_aware_research_ready"] = False
        body = dict(envelope)
        body.pop("envelope_sha256")
        envelope["envelope_sha256"] = canonical_json_sha256(body)
    with pytest.raises(SafeSuccessorError, match="SUCCESSOR_MACHINE_STATE_MISMATCH"):
        validate_successor_provenance(
            altered_pointer,
            altered_manifest,
            generation_root=capture.staging_root,
            historical_only=True,
        )

    capture.pointer_path.write_bytes(b'{"status":"tampered"}\n')
    with pytest.raises(SafeSuccessorError, match="SUCCESSOR_JSON_READBACK_MISMATCH"):
        validate_successor_provenance(
            pointer,
            manifest,
            generation_root=capture.staging_root,
            historical_only=True,
        )
    capture.pointer_path.write_bytes(capture.pointer_bytes)
    capture.manifest_path.write_bytes(b'{"status":"tampered"}\n')
    with pytest.raises(SafeSuccessorError, match="SUCCESSOR_JSON_READBACK_MISMATCH"):
        validate_successor_provenance(
            pointer,
            manifest,
            generation_root=capture.staging_root,
            historical_only=True,
        )
    capture.manifest_path.write_bytes(capture.manifest_bytes)

    live_market_pointer = Path(targets["market_pointer"]["path"])
    original_live_market = live_market_pointer.read_bytes()
    live_market_pointer.write_bytes(b'{"snapshot_id":"advanced-live-pointer"}\n')
    validate_successor_provenance(
        pointer,
        manifest,
        generation_root=capture.staging_root,
        historical_only=True,
    )
    with pytest.raises(SafeSuccessorError, match="TARGET_EVIDENCE_TAMPER"):
        validate_successor_provenance(
            pointer,
            manifest,
            generation_root=capture.staging_root,
            historical_only=False,
        )
    live_market_pointer.write_bytes(original_live_market)
    assert capture.table_sha256["fundamental_quarantine"] == (
        bundle.predecessor_binding["table_sha256"]["fundamental_quarantine"]
    )
    evidence_root = (
        capture.staging_root
        / "_fundamental_generations"
        / capture.generation_id
        / "provider_evidence"
    )
    predecessor_pointer = evidence_root / "sealed/predecessor_pointer.json"
    original = predecessor_pointer.read_bytes()
    predecessor_pointer.write_bytes(b"tampered pointer\n")
    with pytest.raises(SafeSuccessorError, match="SUPPORT_REFERENCE_TAMPER"):
        validate_successor_provenance(
            pointer,
            manifest,
            generation_root=capture.staging_root,
            historical_only=True,
        )
    predecessor_pointer.write_bytes(original)

    for name in ("market_pointer", "pit_pointer"):
        relative = Path(
            pointer["metadata"]["target_bindings"][name]["sealed_ref"]["path"]
        )
        sealed_pointer = evidence_root / relative
        original_sealed = sealed_pointer.read_bytes()
        sealed_pointer.write_bytes(b'{"tampered":true}\n')
        with pytest.raises(SafeSuccessorError, match="TARGET_SEALED_REF_TAMPER"):
            validate_successor_provenance(
                pointer,
                manifest,
                generation_root=capture.staging_root,
                historical_only=True,
            )
        sealed_pointer.write_bytes(original_sealed)

    support_manifest = evidence_root / "sealed/support_manifest.json"
    support_manifest.write_bytes(b"tampered support manifest\n")
    with pytest.raises(SafeSuccessorError, match="SUPPORT_REFERENCE_TAMPER"):
        validate_successor_provenance(
            pointer,
            manifest,
            generation_root=capture.staging_root,
            historical_only=True,
        )


def test_stage_and_provenance_replay_same_versioned_source_fileset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, targets, support_files, _provider = _path_backed_case(tmp_path)
    support_files = dict(support_files)
    support_files["source/binding.json"] = b'{"binding":"source"}\n'
    source_manifest_sha256 = "a" * 64
    implementation_sha256 = "b" * 64
    provider = seal_successor_provider_manifest(
        bundle,
        provider="fixture",
        request_receipts_sha256="f" * 64,
        evidence_files={name: _sha(payload) for name, payload in support_files.items()},
        extra={
            "implementation_sha256": implementation_sha256,
            "source_fileset_manifest_sha256": source_manifest_sha256,
        },
    )
    calls: list[Path] = []

    def validate_source(
        root: str | Path,
        *,
        expected_implementation_sha256: str | None = None,
    ) -> dict[str, object]:
        source_root = Path(root)
        calls.append(source_root)
        assert (source_root / "binding.json").read_bytes() == support_files[
            "source/binding.json"
        ]
        assert expected_implementation_sha256 == implementation_sha256
        return {
            "manifest_sha256": source_manifest_sha256,
            "resource_accounting": {"resource_sha256": "c" * 64},
            "schema_version": "cn-fundamental-successor-provider-manifest.v3",
        }

    monkeypatch.setattr(
        "quant_investor.market.fundamental_successor_source."
        "validate_successor_support_fileset",
        validate_source,
    )
    capture = stage_successor_generation(
        bundle,
        staging_root=tmp_path / "staging-with-source",
        generation_id="staged_successor",
        provider_manifest=provider,
        target_bindings=targets,
        provider_evidence_files=support_files,
    )
    pointer = json.loads(capture.pointer_bytes)
    manifest = json.loads(capture.manifest_bytes)
    assert pointer["metadata"]["source_fileset_validation"]["status"] == "PASS"

    receipt = validate_successor_provenance(
        pointer,
        manifest,
        generation_root=capture.staging_root,
        historical_only=True,
    )
    assert receipt["source_fileset_validation"]["manifest_sha256"] == (
        source_manifest_sha256
    )
    # Staging validates once before sealing and once through its own exact
    # readback; the independent provenance call replays the same source policy.
    assert len(calls) == 3


def _append_first_raw(*, include_partial_financial: bool = False) -> dict[str, pd.DataFrame]:
    raw = {name: pd.DataFrame() for name in RAW_TABLES}
    if include_partial_financial:
        raw["income"] = pd.DataFrame(
            [
                _financial_row(
                    "income",
                    "000001.SZ",
                    "20260630",
                    "20260807",
                    delta=True,
                )
            ]
        )
    raw["daily_basic"] = pd.DataFrame(
        [
            {
                "ts_code": symbol,
                "trade_date": "20260807",
                "total_mv": value,
                "sector": "fixture-sector",
            }
            for symbol, value in zip(SYMBOLS, ("100", "200", "300"))
        ]
    )
    return raw


def test_append_first_uses_immutable_parent_without_prefix_replay() -> None:
    parents = _parent_tables()
    raw = _append_first_raw()
    registry_sha = "9" * 64
    plan = seal_support_plan(
        raw,
        parent_cutoff=PARENT_CUTOFF,
        target_cutoff="20260807",
        permanent_support_refs=_support_refs(),
        support_prefix_mode=SUCCESSOR_APPEND_FIRST_MODE,
        historical_taint_registry_sha256=registry_sha,
        extra={"support_start": "20260807"},
    )
    bundle = assemble_safe_successor(
        parent_tables=parents,
        parent_closure=_parent_closure(parents),
        support_raw_tables=raw,
        plan_metadata=plan,
        keyset_closure=_keyset(raw),
        parent_cutoff=PARENT_CUTOFF,
        target_cutoff="20260807",
        run_id="append_first_successor",
    )

    assert bundle.derivation_evidence["support_prefix_mode"] == (
        SUCCESSOR_APPEND_FIRST_MODE
    )
    assert bundle.derivation_evidence["historical_taint_registry_sha256"] == (
        registry_sha
    )
    assert bundle.lineage["boundary"]["support_prefix_row_count"] == 0
    assert len(bundle.period_suffix) == 0
    assert len(bundle.daily_suffix) == len(SYMBOLS)


def test_append_first_blocks_delta_with_unclosed_historical_dependency() -> None:
    parents = _parent_tables()
    raw = _append_first_raw(include_partial_financial=True)
    plan = seal_support_plan(
        raw,
        parent_cutoff=PARENT_CUTOFF,
        target_cutoff="20260807",
        permanent_support_refs=_support_refs(),
        support_prefix_mode=SUCCESSOR_APPEND_FIRST_MODE,
        historical_taint_registry_sha256="9" * 64,
        extra={"support_start": "20260807"},
    )

    with pytest.raises(SafeSuccessorError, match="SUPPORT_START_ANCHOR_UNCLOSED"):
        assemble_safe_successor(
            parent_tables=parents,
            parent_closure=_parent_closure(parents),
            support_raw_tables=raw,
            plan_metadata=plan,
            keyset_closure=_keyset(raw),
            parent_cutoff=PARENT_CUTOFF,
            target_cutoff="20260807",
            run_id="append_first_blocked",
        )


def test_append_first_consumes_only_declared_income_support_dependency() -> None:
    parents = _parent_tables()
    raw = _append_first_raw(include_partial_financial=True)
    raw["income"] = pd.concat(
        [
            pd.DataFrame(
                [
                    _financial_row(
                        "income",
                        "000001.SZ",
                        "20250630",
                        "20250830",
                        delta=False,
                    )
                ]
            ),
            raw["income"],
        ],
        ignore_index=True,
    )
    indicator = _financial_row(
        "fina_indicator",
        "000001.SZ",
        "20260630",
        "20260807",
        delta=True,
    )
    indicator["netprofit_yoy"] = np.nan
    raw["fina_indicator"] = pd.DataFrame([indicator])
    raw["balancesheet"] = pd.DataFrame(
        [
            _financial_row(
                "balancesheet",
                "000001.SZ",
                "20260630",
                "20260807",
                delta=True,
            )
        ]
    )
    raw["cashflow"] = pd.DataFrame(
        [
            _financial_row(
                "cashflow",
                "000001.SZ",
                "20260630",
                "20260807",
                delta=True,
            )
        ]
    )
    dependency = {"ts_code": "000001.SZ", "end_date": "20250630"}
    plan = seal_support_plan(
        raw,
        parent_cutoff=PARENT_CUTOFF,
        target_cutoff="20260807",
        permanent_support_refs=_support_refs(),
        support_prefix_mode=SUCCESSOR_APPEND_FIRST_MODE,
        historical_taint_registry_sha256="9" * 64,
        append_first_income_dependencies=(dependency,),
        extra={"support_start": "20260807"},
    )

    bundle = assemble_safe_successor(
        parent_tables=parents,
        parent_closure=_parent_closure(parents),
        support_raw_tables=raw,
        plan_metadata=plan,
        keyset_closure=_keyset(raw),
        parent_cutoff=PARENT_CUTOFF,
        target_cutoff="20260807",
        run_id="append_first_dependency",
    )

    assert len(bundle.period_suffix) == 1
    assert bundle.lineage["boundary"]["support_prefix_row_count"] == 0
    assert bundle.lineage["period_rows"][0]["previous_year_income"]["row_binding"]


def test_append_first_consumes_declared_balancesheet_support_dependency() -> None:
    parents = _parent_tables()
    raw = _append_first_raw()
    indicator = _financial_row(
        "fina_indicator",
        "000001.SZ",
        "20260630",
        "20260807",
        delta=True,
    )
    indicator["debt_to_assets"] = np.nan
    raw["fina_indicator"] = pd.DataFrame([indicator])
    raw["income"] = pd.DataFrame(
        [
            _financial_row(
                "income",
                "000001.SZ",
                "20260630",
                "20260807",
                delta=True,
            )
        ]
    )
    raw["cashflow"] = pd.DataFrame(
        [
            _financial_row(
                "cashflow",
                "000001.SZ",
                "20260630",
                "20260807",
                delta=True,
            )
        ]
    )
    raw["balancesheet"] = pd.DataFrame(
        [
            _financial_row(
                "balancesheet",
                "000001.SZ",
                "20260630",
                "20260801",
                delta=False,
            )
        ]
    )
    plan = seal_support_plan(
        raw,
        parent_cutoff=PARENT_CUTOFF,
        target_cutoff="20260807",
        permanent_support_refs=_support_refs(),
        support_prefix_mode=SUCCESSOR_APPEND_FIRST_MODE,
        historical_taint_registry_sha256="9" * 64,
        append_first_financial_dependencies=(
            {
                "table": "balancesheet",
                "ts_code": "000001.SZ",
                "end_date": "20260630",
            },
        ),
        extra={"support_start": "20260807"},
    )

    bundle = assemble_safe_successor(
        parent_tables=parents,
        parent_closure=_parent_closure(parents),
        support_raw_tables=raw,
        plan_metadata=plan,
        keyset_closure=_keyset(raw),
        parent_cutoff=PARENT_CUTOFF,
        target_cutoff="20260807",
        run_id="append_first_balance_dependency",
    )

    requirements = bundle.lineage["period_rows"][0]["dependency_requirements"]
    assert any(
        row["table"] == "balancesheet" and row["row_binding"]
        for row in requirements
    )


def test_append_first_rejects_declared_income_support_not_consumed() -> None:
    parents = _parent_tables()
    raw = _append_first_raw()
    raw["income"] = pd.DataFrame(
        [
            _financial_row(
                "income",
                "000001.SZ",
                "20250630",
                "20250830",
                delta=False,
            )
        ]
    )
    plan = seal_support_plan(
        raw,
        parent_cutoff=PARENT_CUTOFF,
        target_cutoff="20260807",
        permanent_support_refs=_support_refs(),
        support_prefix_mode=SUCCESSOR_APPEND_FIRST_MODE,
        historical_taint_registry_sha256="9" * 64,
        append_first_income_dependencies=(
            {"ts_code": "000001.SZ", "end_date": "20250630"},
        ),
        extra={"support_start": "20260807"},
    )

    with pytest.raises(
        SafeSuccessorError,
        match="APPEND_FIRST_FINANCIAL_DEPENDENCY_NOT_EXACTLY_CONSUMED",
    ):
        assemble_safe_successor(
            parent_tables=parents,
            parent_closure=_parent_closure(parents),
            support_raw_tables=raw,
            plan_metadata=plan,
            keyset_closure=_keyset(raw),
            parent_cutoff=PARENT_CUTOFF,
            target_cutoff="20260807",
            run_id="append_first_unused_dependency",
        )


def test_append_first_consumes_exact_provider_absence_proof() -> None:
    parents = _parent_tables()
    raw = _append_first_raw(include_partial_financial=True)
    indicator = _financial_row(
        "fina_indicator",
        "000001.SZ",
        "20260630",
        "20260807",
        delta=True,
    )
    indicator["netprofit_yoy"] = np.nan
    raw["fina_indicator"] = pd.DataFrame([indicator])
    raw["balancesheet"] = pd.DataFrame(
        [
            _financial_row(
                "balancesheet",
                "000001.SZ",
                "20260630",
                "20260807",
                delta=True,
            )
        ]
    )
    raw["cashflow"] = pd.DataFrame(
        [
            _financial_row(
                "cashflow",
                "000001.SZ",
                "20260630",
                "20260807",
                delta=True,
            )
        ]
    )
    plan = seal_support_plan(
        raw,
        parent_cutoff=PARENT_CUTOFF,
        target_cutoff="20260807",
        permanent_support_refs=_support_refs(),
        support_prefix_mode=SUCCESSOR_APPEND_FIRST_MODE,
        historical_taint_registry_sha256="9" * 64,
        absence_proofs=(
            {
                "status": "PROVEN_ABSENT",
                "table": "income",
                "symbol": "000001.SZ",
                "end_date": "20250630",
                "available_through": "20260807",
                "evidence_sha256": "8" * 64,
            },
        ),
        extra={"support_start": "20260807"},
    )

    bundle = assemble_safe_successor(
        parent_tables=parents,
        parent_closure=_parent_closure(parents),
        support_raw_tables=raw,
        plan_metadata=plan,
        keyset_closure=_keyset(raw),
        parent_cutoff=PARENT_CUTOFF,
        target_cutoff="20260807",
        run_id="append_first_absence",
    )

    assert bundle.lineage["period_rows"][0]["previous_year_income"]["absence_proven"]
    assert bundle.lineage["boundary"]["bounded_financial_absence_proof_count"] == 1


def test_append_first_rejects_unused_provider_absence_proof() -> None:
    parents = _parent_tables()
    raw = _append_first_raw()
    plan = seal_support_plan(
        raw,
        parent_cutoff=PARENT_CUTOFF,
        target_cutoff="20260807",
        permanent_support_refs=_support_refs(),
        support_prefix_mode=SUCCESSOR_APPEND_FIRST_MODE,
        historical_taint_registry_sha256="9" * 64,
        absence_proofs=(
            {
                "status": "PROVEN_ABSENT",
                "table": "income",
                "symbol": "000001.SZ",
                "end_date": "20250630",
                "available_through": "20260807",
                "evidence_sha256": "8" * 64,
            },
        ),
        extra={"support_start": "20260807"},
    )

    with pytest.raises(
        SafeSuccessorError,
        match="APPEND_FIRST_FINANCIAL_ABSENCE_NOT_EXACTLY_CONSUMED",
    ):
        assemble_safe_successor(
            parent_tables=parents,
            parent_closure=_parent_closure(parents),
            support_raw_tables=raw,
            plan_metadata=plan,
            keyset_closure=_keyset(raw),
            parent_cutoff=PARENT_CUTOFF,
            target_cutoff="20260807",
            run_id="append_first_unused_absence",
        )
