from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from quant_investor.market import fundamental_mart


def _audit_outcome(
    symbol: str,
    table: str,
    status: str,
    **extra: object,
) -> dict[str, object]:
    rows = 1 if status == "success" else 0
    outcome: dict[str, object] = {
        "schema_version": fundamental_mart.FUNDAMENTAL_REQUEST_OUTCOME_SCHEMA,
        "symbol": symbol,
        "table": table,
        "status": status,
        "rows_received": rows,
        "rows": rows,
        "rows_hard_invalid": 0,
        "rows_filtered_future": 0,
        "rows_filtered_missing_availability": 0,
        "rows_filtered_core_values": 0,
        "rows_deduplicated": 0,
        "rows_discarded_request_malformed": 0,
        "rows_hard_invalid_schema": 0,
        "rows_hard_invalid_symbol": 0,
        "rows_hard_invalid_availability_date": 0,
        "rows_hard_invalid_end_date": 0,
        "rows_hard_invalid_end_after_availability": 0,
        "rows_hard_invalid_core_numeric": 0,
        **extra,
    }
    if table in fundamental_mart.FINANCIAL_SOURCE_TABLES and status == "success":
        outcome["financial_coverage"] = {
            "status": "not_applicable",
            "passed": True,
        }
        outcome["financial_coverage_passed"] = True
    return outcome


class _Provider:
    def __init__(self, *, daily_empty: bool = False, retry_fina: int = 0) -> None:
        self.calls: list[tuple[str, str]] = []
        self.daily_empty = daily_empty
        self.retry_fina = retry_fina

    def __getattr__(self, table: str):
        if table not in fundamental_mart.SOURCE_TABLES:
            raise AttributeError(table)

        def fetch(**kwargs):
            symbol = kwargs["ts_code"]
            self.calls.append((table, symbol))
            if table == "fina_indicator" and self.retry_fina > 0:
                self.retry_fina -= 1
                raise RuntimeError("transient quota")
            if table == "forecast" or (table == "daily_basic" and self.daily_empty):
                return pd.DataFrame()
            if table == "daily_basic":
                return pd.DataFrame(
                    [
                        {
                            "ts_code": symbol,
                            "trade_date": "20240510",
                            "total_mv": 1.0,
                            "circ_mv": 1.0,
                            "pe": 10.0,
                            "pb": 1.0,
                        }
                    ]
                )
            values = {
                "fina_indicator": {
                    "roe_dt": 10.0,
                    "roe": 10.0,
                    "roa": 5.0,
                    "debt_to_assets": 40.0,
                    "netprofit_yoy": 5.0,
                    "ocf_to_profit": 1.0,
                },
                "income": {"n_income": 1.0, "n_income_attr_p": 1.0},
                "balancesheet": {"total_liab": 1.0, "total_assets": 2.0},
                "cashflow": {
                    "n_cashflow_act": 1.0,
                    "c_pay_acq_const_fiolta": 0.1,
                    "free_cashflow": 0.9,
                },
            }[table]
            return pd.DataFrame(
                [
                    {
                        "ts_code": symbol,
                        "ann_date": "20240430",
                        "end_date": "20231231",
                        "update_flag": "0",
                        **values,
                    }
                ]
            )

        return fetch


class _CoverageProvider:
    def __init__(
        self,
        *,
        income_complete: bool = True,
        daily_complete: bool = True,
        marker: float = 1.0,
    ) -> None:
        self.calls: list[tuple[str, str]] = []
        self.income_complete = income_complete
        self.daily_complete = daily_complete
        self.marker = marker

    def __getattr__(self, table: str):
        if table not in fundamental_mart.SOURCE_TABLES:
            raise AttributeError(table)

        def fetch(**kwargs):
            symbol = kwargs["ts_code"]
            self.calls.append((table, symbol))
            if table == "forecast":
                return pd.DataFrame()
            if table == "daily_basic":
                dates = (
                    pd.bdate_range("2020-01-01", "2024-05-10")
                    if self.daily_complete
                    else pd.DatetimeIndex(
                        [pd.Timestamp("2020-01-12"), pd.Timestamp("2024-05-10")]
                    )
                )
                return pd.DataFrame(
                    {
                        "ts_code": symbol,
                        "trade_date": dates.strftime("%Y%m%d"),
                        "total_mv": self.marker,
                        "circ_mv": self.marker,
                        "pe": 10.0,
                        "pb": 1.0,
                    }
                )
            periods = pd.period_range("2020Q1", "2023Q4", freq="Q-DEC")
            end_dates = pd.DatetimeIndex(
                [period.end_time.normalize() for period in periods]
            )
            if table == "income" and not self.income_complete:
                end_dates = end_dates[:-1]
            records: list[dict[str, object]] = []
            for end_date in end_dates:
                values = {
                    "fina_indicator": {
                        "roe_dt": self.marker,
                        "roe": self.marker,
                        "roa": self.marker,
                        "debt_to_assets": self.marker,
                        "netprofit_yoy": self.marker,
                    },
                    "income": {
                        "n_income": self.marker,
                        "n_income_attr_p": self.marker,
                        "update_flag": "0",
                    },
                    "balancesheet": {
                        "total_liab": self.marker,
                        "total_assets": self.marker + 1.0,
                        "update_flag": "0",
                    },
                    "cashflow": {
                        "n_cashflow_act": self.marker,
                        "c_pay_acq_const_fiolta": 0.1,
                        "free_cashflow": self.marker,
                        "update_flag": "0",
                    },
                }[table]
                records.append(
                    {
                        "ts_code": symbol,
                        "ann_date": (end_date + pd.Timedelta(days=60)).strftime(
                            "%Y%m%d"
                        ),
                        "end_date": end_date.strftime("%Y%m%d"),
                        **values,
                    }
                )
            return pd.DataFrame(records)

        return fetch


def _scope_file(tmp_path: Path) -> Path:
    path = tmp_path / "cn_index_components.json"
    path.write_text(json.dumps({"full_a": ["000001.SZ"]}), encoding="utf-8")
    return path


def _market_pointer_file(
    tmp_path: Path,
    *,
    as_of: str = "20240510",
    non_blocking_absent_symbols: list[str] | None = None,
    membership_path: Path | None = None,
) -> Path:
    path = tmp_path / "_latest.json"
    scope_sha = fundamental_mart._symbol_scope_sha256(["000001.SZ"])
    resolved_membership = membership_path or _membership_file(tmp_path)
    membership = pd.read_parquet(resolved_membership)
    bars_root = tmp_path / "canonical-bars"
    bars_root.mkdir(exist_ok=True)
    as_of_ts = pd.Timestamp(pd.to_datetime(as_of, format="%Y%m%d"))
    rows: list[dict[str, str]] = []
    for membership_row in membership.to_dict("records"):
        symbol = str(membership_row["symbol"])
        list_date_value = pd.to_datetime(
            str(membership_row["list_date"]),
            format="%Y%m%d",
            errors="coerce",
        )
        list_date = (
            pd.Timestamp(list_date_value)
            if not pd.isna(list_date_value)
            else as_of_ts - pd.DateOffset(years=5)
        )
        end_values: list[pd.Timestamp] = []
        for field in ("effective_to", "delist_date"):
            value = str(membership_row.get(field) or "").strip()
            if value:
                parsed_end = pd.to_datetime(
                    value,
                    format="%Y%m%d",
                    errors="coerce",
                )
                if not pd.isna(parsed_end):
                    end_values.append(pd.Timestamp(parsed_end))
        history_start = max(as_of_ts - pd.DateOffset(years=5), list_date)
        history_end = min(as_of_ts, max(end_values)) if end_values else as_of_ts
        dates = pd.bdate_range(history_start, history_end)
        if dates.empty:
            dates = pd.DatetimeIndex([history_end])
        rows.extend(
            {
                "ts_code": symbol,
                "trade_date": trade_date.strftime("%Y%m%d"),
            }
            for trade_date in dates
        )
    pd.DataFrame(rows).to_parquet(bars_root / "part.parquet", index=False)
    path.write_text(
        json.dumps(
            {
                "snapshot_id": "scope-test",
                "latest_complete_trade_date": as_of,
                "table_root": str(bars_root.resolve()),
                "coverage": {
                    "expected_scope_count": 1,
                    "expected_scope_sha256": scope_sha,
                    "non_blocking_absent_symbols": list(
                        non_blocking_absent_symbols or []
                    ),
                    "pit_membership_path": str(resolved_membership.resolve()),
                    "pit_membership_sha256": hashlib.sha256(
                        resolved_membership.read_bytes()
                    ).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _membership_file(tmp_path: Path) -> Path:
    path = tmp_path / "stock_basic_membership.parquet"
    if path.exists():
        return path
    pd.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "list_date": "20240510",
                "effective_from": "20240510",
                "effective_to": "",
                "delist_date": "",
            }
        ]
    ).to_parquet(path, index=False)
    return path


def _long_membership_file(tmp_path: Path) -> Path:
    path = tmp_path / "long_stock_basic_membership.parquet"
    pd.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "list_date": "20200101",
                "effective_from": "20200101",
                "effective_to": "",
                "delist_date": "",
            }
        ]
    ).to_parquet(path, index=False)
    return path


def test_live_fetch_applies_strict_pit_cutoff() -> None:
    class _FutureProvider(_Provider):
        def __getattr__(self, table: str):
            base = super().__getattr__(table)

            def fetch(**kwargs):
                frame = base(**kwargs)
                if frame.empty:
                    return frame
                future = frame.copy()
                date_column = "trade_date" if table == "daily_basic" else "ann_date"
                future[date_column] = "20240511"
                return pd.concat([frame, future], ignore_index=True)

            return fetch

    tables, manifest = fundamental_mart._fetch_tushare_tables(
        ["000001.SZ"],
        years=5,
        as_of="20240510",
        workers=1,
        pro=_FutureProvider(),
        symbol_pause_seconds=0,
    )

    assert len(tables["daily_basic"]) == 1
    assert len(tables["fina_indicator"]) == 1
    assert manifest["pit_rows_filtered_future"] == 5
    assert all(str(value) <= "20240510" for value in tables["daily_basic"]["trade_date"])


def test_fina_indicator_request_uses_provider_supported_fields() -> None:
    fields = set(fundamental_mart.SOURCE_REQUEST_FIELDS["fina_indicator"].split(","))

    assert "ocf_to_profit" not in fields
    assert "update_flag" not in fields
    assert {"ts_code", "ann_date", "end_date", "roe_dt", "netprofit_yoy"}.issubset(
        fields
    )


@pytest.mark.parametrize(
    ("table", "date_column", "malformed_date", "hard_counter", "expected_reason"),
    [
        (
            "daily_basic",
            "trade_date",
            "20240510junk",
            "rows_hard_invalid_availability_date",
            "invalid_availability_date",
        ),
        (
            "fina_indicator",
            "ann_date",
            "2024051020260510",
            "rows_hard_invalid_availability_date",
            "invalid_availability_date",
        ),
        (
            "income",
            "end_date",
            "not-a-date",
            "rows_hard_invalid_end_date",
            "invalid_end_date",
        ),
    ],
)
def test_strict_pit_cutoff_rejects_date_suffixes(
    table: str,
    date_column: str,
    malformed_date: str,
    hard_counter: str,
    expected_reason: str,
) -> None:
    frame = getattr(_Provider(), table)(ts_code="000001.SZ")
    frame[date_column] = malformed_date

    accepted, stats, reason = fundamental_mart._strict_pit_cutoff(
        frame,
        table=table,
        symbol="000001.SZ",
        as_of="20240510",
    )

    assert accepted.empty
    assert stats[hard_counter] == 1
    assert stats["rows_discarded_request_malformed"] == 0
    assert reason == expected_reason


def test_strict_pit_cutoff_rejects_financial_end_after_availability() -> None:
    frame = _Provider().fina_indicator(ts_code="000001.SZ")
    frame["end_date"] = "20240501"
    frame["ann_date"] = "20240430"

    accepted, stats, reason = fundamental_mart._strict_pit_cutoff(
        frame,
        table="fina_indicator",
        symbol="000001.SZ",
        as_of="20240510",
    )

    assert accepted.empty
    assert stats["rows_hard_invalid_end_after_availability"] == 1
    assert reason == "end_after_availability"


def test_strict_pit_cutoff_allows_forecast_period_after_announcement() -> None:
    frame = _Provider().forecast(ts_code="000001.SZ")
    assert frame.empty
    frame = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "ann_date": "20240430",
                "end_date": "20240630",
                "type": "预增",
                "p_change_min": 10.0,
                "p_change_max": 20.0,
                "net_profit_min": 1.0,
                "net_profit_max": 2.0,
                "last_parent_net": 1.0,
                "summary": "fixture",
                "change_reason": "fixture",
                "update_flag": "0",
            }
        ]
    )

    accepted, stats, reason = fundamental_mart._strict_pit_cutoff(
        frame,
        table="forecast",
        symbol="000001.SZ",
        as_of="20240510",
    )

    assert len(accepted) == 1
    assert stats["rows_hard_invalid"] == 0
    assert reason == ""


def test_missing_availability_row_is_filtered_without_poisoning_valid_row() -> None:
    valid = _Provider().fina_indicator(ts_code="000001.SZ")
    missing = valid.copy()
    missing["ann_date"] = pd.NA
    frame = pd.concat([valid, missing], ignore_index=True)

    accepted, stats, reason = fundamental_mart._strict_pit_cutoff(
        frame,
        table="fina_indicator",
        symbol="000001.SZ",
        as_of="20240510",
    )

    assert reason == ""
    assert len(accepted) == 1
    assert stats["rows_filtered_missing_availability"] == 1
    assert stats["rows_hard_invalid"] == 0
    fundamental_mart.validate_outcome_accounting_v3(
        {
            "schema_version": fundamental_mart.FUNDAMENTAL_REQUEST_OUTCOME_SCHEMA,
            "status": "success",
            **stats,
        }
    )


def test_populated_invalid_availability_discards_otherwise_valid_response() -> None:
    valid = _Provider().fina_indicator(ts_code="000001.SZ")
    invalid = valid.copy()
    invalid["ann_date"] = "20240510junk"

    accepted, stats, reason = fundamental_mart._strict_pit_cutoff(
        pd.concat([valid, invalid], ignore_index=True),
        table="fina_indicator",
        symbol="000001.SZ",
        as_of="20240510",
    )

    assert accepted.empty
    assert reason == "invalid_availability_date"
    assert stats["rows_hard_invalid_availability_date"] == 1
    assert stats["rows_discarded_request_malformed"] == 1
    assert stats["rows"] == 0


def test_core_empty_row_filters_but_nonfinite_value_hard_fails() -> None:
    valid = _Provider().fina_indicator(ts_code="000001.SZ")
    empty = valid.copy()
    for column in ("roe_dt", "roe", "roa", "debt_to_assets", "netprofit_yoy"):
        empty[column] = pd.NA
    accepted, stats, reason = fundamental_mart._strict_pit_cutoff(
        pd.concat([valid, empty], ignore_index=True),
        table="fina_indicator",
        symbol="000001.SZ",
        as_of="20240510",
    )
    assert reason == ""
    assert len(accepted) == 1
    assert stats["rows_filtered_core_values"] == 1

    nonfinite = valid.copy()
    nonfinite["roe"] = np.inf
    accepted, stats, reason = fundamental_mart._strict_pit_cutoff(
        pd.concat([valid, nonfinite], ignore_index=True),
        table="fina_indicator",
        symbol="000001.SZ",
        as_of="20240510",
    )
    assert accepted.empty
    assert reason == "invalid_core_values"
    assert stats["rows_hard_invalid_core_numeric"] == 1
    assert stats["rows_discarded_request_malformed"] == 1


def test_daily_basic_requires_positive_total_mv_not_circ_mv_only() -> None:
    frame = _Provider().daily_basic(ts_code="000001.SZ")
    frame["total_mv"] = pd.NA
    frame["circ_mv"] = 100.0

    accepted, stats, reason = fundamental_mart._strict_pit_cutoff(
        frame,
        table="daily_basic",
        symbol="000001.SZ",
        as_of="20240510",
    )

    assert reason == ""
    assert accepted.empty
    assert stats["rows_filtered_core_values"] == 1


def test_endpoint_audit_marks_one_invalid_financial_request_malformed() -> None:
    symbols = [f"{index:06d}.SZ" for index in range(1, 101)]

    class _InvalidIncomeDateProvider(_Provider):
        def __getattr__(self, table: str):
            base = super().__getattr__(table)

            def fetch(**kwargs):
                frame = base(**kwargs)
                if (
                    table == "income"
                    and kwargs["ts_code"] == symbols[-1]
                    and not frame.empty
                ):
                    frame["ann_date"] = "20240510junk"
                return frame

            return fetch

    with pytest.raises(fundamental_mart.FundamentalFetchAuditError) as exc_info:
        fundamental_mart._fetch_tushare_tables(
            symbols,
            years=0,
            as_of="20240510",
            workers=1,
            pro=_InvalidIncomeDateProvider(),
            enforce_endpoint_audit=True,
            symbol_pause_seconds=0,
        )

    income = exc_info.value.manifest["endpoint_audit"]["endpoints"]["income"]
    assert income["success"] == 99
    assert income["malformed"] == 1
    assert "provider_malformed_requests_above_threshold" in exc_info.value.manifest[
        "endpoint_audit"
    ]["blockers"]


def test_live_fetch_retries_with_bounded_attempts() -> None:
    provider = _Provider(retry_fina=2)

    _tables, manifest = fundamental_mart._fetch_tushare_tables(
        ["000001.SZ"],
        years=5,
        as_of="20240510",
        workers=1,
        pro=provider,
        max_attempts=3,
        retry_backoff_seconds=0,
        symbol_pause_seconds=0,
    )

    outcome = next(
        item for item in manifest["symbol_table_outcomes"] if item["table"] == "fina_indicator"
    )
    assert outcome["status"] == "success"
    assert outcome["attempts"] == 3
    assert manifest["requests_retried"] == 2


def test_full_rebuild_checkpoint_resumes_without_refetch(tmp_path: Path) -> None:
    scope_path = _scope_file(tmp_path)
    checkpoint_root = tmp_path / "checkpoint"
    first_provider = _Provider()

    first_tables, first_manifest = fundamental_mart.fetch_tushare_fundamental_full_rebuild(
        ["000001.SZ"],
        canonical_scope_path=scope_path,
        canonical_market_pointer_path=_market_pointer_file(tmp_path),
        canonical_membership_path=_membership_file(tmp_path),
        years=5,
        as_of="20240510",
        workers=1,
        pro=first_provider,
        checkpoint_root=checkpoint_root,
        requests_per_second=0,
        retry_backoff_seconds=0,
    )
    second_provider = _Provider()
    second_tables, second_manifest = fundamental_mart.fetch_tushare_fundamental_full_rebuild(
        ["000001.SZ"],
        canonical_scope_path=scope_path,
        canonical_market_pointer_path=_market_pointer_file(tmp_path),
        canonical_membership_path=_membership_file(tmp_path),
        years=5,
        as_of="20240510",
        workers=1,
        pro=second_provider,
        checkpoint_root=checkpoint_root,
        requests_per_second=0,
        retry_backoff_seconds=0,
    )

    assert len(first_provider.calls) == 6
    assert second_provider.calls == []
    assert second_manifest["checkpoint"]["resumed_valid_request_count"] == 6
    assert second_manifest["checkpoint"]["requests_fetched_this_run"] == 0
    assert len(second_tables["daily_basic"]) == len(first_tables["daily_basic"])
    assert first_manifest["canonical_scope_evidence"]["canonical_path"] == str(scope_path.resolve())


def test_checkpoint_resume_refetch_policy_covers_audit_failures() -> None:
    symbol = "000001.SZ"
    applicable_gap = _audit_outcome(symbol, "income", "success")
    applicable_gap.update(
        financial_coverage={"status": "applicable", "passed": False},
        financial_coverage_passed=False,
    )
    not_applicable = _audit_outcome(symbol, "income", "success")

    assert fundamental_mart._checkpoint_outcome_requires_refetch(
        _audit_outcome(symbol, "forecast", "malformed"),
        daily_basic_empty_exception_symbols=(),
    )
    assert fundamental_mart._checkpoint_outcome_requires_refetch(
        applicable_gap,
        daily_basic_empty_exception_symbols=(),
    )
    assert fundamental_mart._checkpoint_outcome_requires_refetch(
        _audit_outcome(symbol, "income", "empty"),
        daily_basic_empty_exception_symbols=(),
    )
    assert not fundamental_mart._checkpoint_outcome_requires_refetch(
        not_applicable,
        daily_basic_empty_exception_symbols=(),
    )
    assert fundamental_mart._checkpoint_outcome_requires_refetch(
        _audit_outcome(
            symbol,
            "daily_basic",
            "success",
            history_complete=False,
        ),
        daily_basic_empty_exception_symbols=(symbol,),
    )
    assert fundamental_mart._checkpoint_outcome_requires_refetch(
        _audit_outcome(symbol, "daily_basic", "empty"),
        daily_basic_empty_exception_symbols=(),
    )
    assert not fundamental_mart._checkpoint_outcome_requires_refetch(
        _audit_outcome(symbol, "daily_basic", "empty"),
        daily_basic_empty_exception_symbols=(symbol,),
    )


def test_checkpoint_publish_rejects_stale_cas_without_pointer_change(
    tmp_path: Path,
) -> None:
    scope_path = _scope_file(tmp_path)
    membership_path = _membership_file(tmp_path)
    market_pointer = _market_pointer_file(tmp_path, membership_path=membership_path)
    checkpoint_root = tmp_path / "checkpoint"
    fundamental_mart.fetch_tushare_fundamental_full_rebuild(
        ["000001.SZ"],
        canonical_scope_path=scope_path,
        canonical_market_pointer_path=market_pointer,
        canonical_membership_path=membership_path,
        years=5,
        as_of="20240510",
        workers=1,
        pro=_Provider(),
        checkpoint_root=checkpoint_root,
        requests_per_second=0,
        retry_backoff_seconds=0,
    )
    pointer_before = (checkpoint_root / "latest.json").read_bytes()
    pointer = json.loads(pointer_before)
    manifest = json.loads((checkpoint_root / pointer["manifest_path"]).read_text())
    state = fundamental_mart._load_fetch_checkpoint(
        checkpoint_root,
        expected_binding=manifest["binding"],
    )

    with pytest.raises(
        fundamental_mart.FundamentalFetchCheckpointError,
        match="pointer CAS mismatch",
    ):
        fundamental_mart._write_fetch_checkpoint(
            checkpoint_root,
            binding=manifest["binding"],
            tables=state.tables,
            outcomes=state.outcomes,
            expected_pointer_sha256="0" * 64,
            expected_revision=state.revision,
        )

    assert (checkpoint_root / "latest.json").read_bytes() == pointer_before


def test_checkpoint_publish_reads_each_table_once_and_load_revalidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_root = fundamental_mart._safe_checkpoint_root(
        tmp_path / "checkpoint"
    )
    binding = {"scope": "readback-count"}
    daily = pd.DataFrame(
        [{"ts_code": "000001.SZ", "trade_date": "20240510", "total_mv": 1.0}]
    )
    tables = {
        table: (daily if table == "daily_basic" else pd.DataFrame())
        for table in fundamental_mart.SOURCE_TABLES
    }
    outcomes = [_audit_outcome("000001.SZ", "daily_basic", "success")]
    parquet_read_count = 0
    full_verify_count = 0
    original_read_parquet = fundamental_mart.pd.read_parquet
    original_verify = fundamental_mart._verify_fetch_checkpoint_pointer_bytes

    def counted_read_parquet(*args, **kwargs):
        nonlocal parquet_read_count
        parquet_read_count += 1
        return original_read_parquet(*args, **kwargs)

    def counted_verify(*args, **kwargs):
        nonlocal full_verify_count
        full_verify_count += 1
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(fundamental_mart.pd, "read_parquet", counted_read_parquet)
    monkeypatch.setattr(
        fundamental_mart,
        "_verify_fetch_checkpoint_pointer_bytes",
        counted_verify,
    )

    published = fundamental_mart._write_fetch_checkpoint(
        checkpoint_root,
        binding=binding,
        tables=tables,
        outcomes=outcomes,
        expected_pointer_sha256="",
        expected_revision=0,
    )

    pointer_bytes = (checkpoint_root / "latest.json").read_bytes()
    assert parquet_read_count == len(fundamental_mart.SOURCE_TABLES)
    assert full_verify_count == 0
    assert hashlib.sha256(pointer_bytes).hexdigest() == published.pointer_sha256
    fundamental_mart.assert_frame_semantics_equal(
        daily,
        published.tables["daily_basic"],
        label="published checkpoint snapshot",
    )

    loaded = fundamental_mart._load_fetch_checkpoint(
        checkpoint_root,
        expected_binding=binding,
    )
    assert parquet_read_count == 2 * len(fundamental_mart.SOURCE_TABLES)
    assert full_verify_count == 1
    assert loaded.pointer_sha256 == published.pointer_sha256


def test_checkpoint_candidate_corruption_leaves_pointer_byte_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope_path = _scope_file(tmp_path)
    membership_path = _membership_file(tmp_path)
    market_pointer = _market_pointer_file(tmp_path, membership_path=membership_path)
    checkpoint_root = tmp_path / "checkpoint"
    fundamental_mart.fetch_tushare_fundamental_full_rebuild(
        ["000001.SZ"],
        canonical_scope_path=scope_path,
        canonical_market_pointer_path=market_pointer,
        canonical_membership_path=membership_path,
        years=5,
        as_of="20240510",
        workers=1,
        pro=_Provider(),
        checkpoint_root=checkpoint_root,
        requests_per_second=0,
        retry_backoff_seconds=0,
    )
    pointer_before = (checkpoint_root / "latest.json").read_bytes()
    pointer = json.loads(pointer_before)
    manifest = json.loads((checkpoint_root / pointer["manifest_path"]).read_text())
    state = fundamental_mart._load_fetch_checkpoint(
        checkpoint_root,
        expected_binding=manifest["binding"],
    )
    original_write = fundamental_mart._atomic_parquet_write

    def corrupt_candidate(path: Path, frame: pd.DataFrame) -> None:
        original_write(path, frame)
        if path.name == "fina_indicator.parquet" and not frame.empty:
            changed = pd.read_parquet(path)
            changed.loc[changed.index[0], "ts_code"] = "999999.SZ"
            changed.to_parquet(path, index=False)

    monkeypatch.setattr(fundamental_mart, "_atomic_parquet_write", corrupt_candidate)
    with pytest.raises(
        fundamental_mart.FundamentalFetchCheckpointError,
        match="semantic readback mismatch",
    ):
        fundamental_mart._write_fetch_checkpoint(
            checkpoint_root,
            binding=manifest["binding"],
            tables=state.tables,
            outcomes=state.outcomes,
            expected_pointer_sha256=state.pointer_sha256,
            expected_revision=state.revision,
        )

    assert (checkpoint_root / "latest.json").read_bytes() == pointer_before


def test_checkpoint_candidate_post_read_tamper_blocks_pointer_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_root = fundamental_mart._safe_checkpoint_root(
        tmp_path / "checkpoint"
    )
    daily = pd.DataFrame(
        [{"ts_code": "000001.SZ", "trade_date": "20240510", "total_mv": 1.0}]
    )
    tables = {
        table: (daily if table == "daily_basic" else pd.DataFrame())
        for table in fundamental_mart.SOURCE_TABLES
    }
    original_write = fundamental_mart._atomic_json_write

    def tamper_after_manifest(path: Path, payload: dict[str, Any]) -> None:
        original_write(path, payload)
        if path.name == "manifest.json":
            daily_path = path.parent / "tables" / "daily_basic.parquet"
            changed = pd.read_parquet(daily_path)
            changed.loc[changed.index[0], "total_mv"] = 999.0
            changed.to_parquet(daily_path, index=False)

    monkeypatch.setattr(
        fundamental_mart,
        "_atomic_json_write",
        tamper_after_manifest,
    )
    with pytest.raises(
        fundamental_mart.FundamentalFetchCheckpointError,
        match="changed before publication",
    ):
        fundamental_mart._write_fetch_checkpoint(
            checkpoint_root,
            binding={"scope": "post-read-tamper"},
            tables=tables,
            outcomes=[_audit_outcome("000001.SZ", "daily_basic", "success")],
            expected_pointer_sha256="",
            expected_revision=0,
        )

    assert not (checkpoint_root / "latest.json").exists()


def test_atomic_parquet_write_does_not_follow_predictable_temp_symlink(
    tmp_path: Path,
) -> None:
    path = tmp_path / "daily_basic.parquet"
    victim = tmp_path / "victim.txt"
    victim.write_bytes(b"do-not-overwrite")
    legacy_temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}"
    )
    legacy_temporary.symlink_to(victim)

    frame = pd.DataFrame([{"ts_code": "000001.SZ", "total_mv": 1.0}])
    fundamental_mart._atomic_parquet_write(path, frame)

    assert victim.read_bytes() == b"do-not-overwrite"
    assert legacy_temporary.is_symlink()
    pd.testing.assert_frame_equal(pd.read_parquet(path), frame)


def test_first_checkpoint_failure_recovers_from_safe_orphan_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_root = fundamental_mart._safe_checkpoint_root(
        tmp_path / "checkpoint"
    )
    binding = {"scope": "test"}
    tables = {
        table: pd.DataFrame() for table in fundamental_mart.SOURCE_TABLES
    }
    original_write = fundamental_mart._atomic_json_write
    fail_latest_once = True

    def fail_first_pointer(path: Path, payload: dict[str, Any]) -> None:
        nonlocal fail_latest_once
        if path.name == "latest.json" and fail_latest_once:
            fail_latest_once = False
            raise OSError("simulated pointer publication failure")
        original_write(path, payload)

    monkeypatch.setattr(
        fundamental_mart,
        "_atomic_json_write",
        fail_first_pointer,
    )

    with pytest.raises(OSError, match="pointer publication failure"):
        fundamental_mart._write_fetch_checkpoint(
            checkpoint_root,
            binding=binding,
            tables=tables,
            outcomes=[],
            expected_pointer_sha256="",
            expected_revision=0,
        )

    assert not (checkpoint_root / "latest.json").exists()
    assert len(list((checkpoint_root / "_generations").iterdir())) == 1
    empty = fundamental_mart._load_fetch_checkpoint(
        checkpoint_root,
        expected_binding=binding,
    )
    assert empty.revision == 0
    assert empty.pointer_sha256 == ""

    recovered = fundamental_mart._write_fetch_checkpoint(
        checkpoint_root,
        binding=binding,
        tables=tables,
        outcomes=[],
        expected_pointer_sha256="",
        expected_revision=0,
    )

    assert recovered.revision == 1
    assert recovered.pointer_sha256
    assert len(list((checkpoint_root / "_generations").iterdir())) == 2


@pytest.mark.parametrize("unsafe_kind", ["unknown_file", "symlink"])
def test_checkpoint_orphan_recovery_rejects_unsafe_generation_state(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    checkpoint_root = fundamental_mart._safe_checkpoint_root(
        tmp_path / "checkpoint"
    )
    generation_root = (
        checkpoint_root / "_generations" / "checkpoint_00000001_123"
    )
    generation_root.mkdir(parents=True)
    if unsafe_kind == "unknown_file":
        (generation_root / "unknown.bin").write_bytes(b"unknown")
    else:
        target = tmp_path / "outside"
        target.mkdir()
        (generation_root / "tables").symlink_to(target, target_is_directory=True)

    with pytest.raises(
        fundamental_mart.FundamentalFetchCheckpointError,
        match="unsafe",
    ):
        fundamental_mart._load_fetch_checkpoint(
            checkpoint_root,
            expected_binding={"scope": "test"},
        )


def test_legacy_v2_checkpoint_rejected_before_provider_call(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint_root.mkdir()
    (checkpoint_root / "latest.json").write_text(
        json.dumps(
            {
                "schema_version": "myquant-fundamental-fetch-checkpoint-pointer.v2",
                "revision": 1,
                "generation_id": "legacy",
                "manifest_path": "missing.json",
                "manifest_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    provider = _Provider()

    with pytest.raises(
        fundamental_mart.FundamentalFetchCheckpointError,
        match="pointer schema mismatch",
    ):
        fundamental_mart.fetch_tushare_fundamental_full_rebuild(
            ["000001.SZ"],
            canonical_scope_path=_scope_file(tmp_path),
            canonical_market_pointer_path=_market_pointer_file(tmp_path),
            canonical_membership_path=_membership_file(tmp_path),
            years=5,
            as_of="20240510",
            workers=1,
            pro=provider,
            checkpoint_root=checkpoint_root,
            requests_per_second=0,
            retry_backoff_seconds=0,
        )

    assert provider.calls == []


def test_checkpoint_rejects_as_of_drift_and_tampering(tmp_path: Path) -> None:
    scope_path = _scope_file(tmp_path)
    checkpoint_root = tmp_path / "checkpoint"
    kwargs: dict[str, Any] = {
        "symbols": ["000001.SZ"],
        "canonical_scope_path": scope_path,
        "canonical_market_pointer_path": _market_pointer_file(tmp_path),
        "canonical_membership_path": _membership_file(tmp_path),
        "years": 5,
        "workers": 1,
        "pro": _Provider(),
        "checkpoint_root": checkpoint_root,
        "requests_per_second": 0,
        "retry_backoff_seconds": 0,
    }
    fundamental_mart.fetch_tushare_fundamental_full_rebuild(as_of="20240510", **kwargs)

    with pytest.raises(
        (fundamental_mart.FundamentalFetchCheckpointError, ValueError),
        match="binding mismatch|does not match as_of",
    ):
        fundamental_mart.fetch_tushare_fundamental_full_rebuild(as_of="20240511", **kwargs)

    pointer = json.loads((checkpoint_root / "latest.json").read_text())
    manifest_path = checkpoint_root / pointer["manifest_path"]
    checkpoint_manifest = json.loads(manifest_path.read_text())
    table_path = manifest_path.parent / checkpoint_manifest["table_files"]["daily_basic"]["path"]
    table_path.write_bytes(table_path.read_bytes() + b"tamper")
    with pytest.raises(
        fundamental_mart.FundamentalFetchCheckpointError,
        match="table SHA mismatch",
    ):
        fundamental_mart.fetch_tushare_fundamental_full_rebuild(as_of="20240510", **kwargs)


def test_resume_preserves_prior_failed_provider_call_accounting(tmp_path: Path) -> None:
    scope_path = _scope_file(tmp_path)
    market_pointer = _market_pointer_file(tmp_path)
    checkpoint_root = tmp_path / "checkpoint"
    with pytest.raises(fundamental_mart.FundamentalFetchAuditError):
        fundamental_mart.fetch_tushare_fundamental_full_rebuild(
            ["000001.SZ"],
            canonical_scope_path=scope_path,
            canonical_market_pointer_path=market_pointer,
            canonical_membership_path=_membership_file(tmp_path),
            years=5,
            as_of="20240510",
            workers=1,
            pro=_Provider(retry_fina=1),
            checkpoint_root=checkpoint_root,
            max_attempts=1,
            requests_per_second=0,
            retry_backoff_seconds=0,
        )

    second_provider = _Provider()
    _tables, manifest = fundamental_mart.fetch_tushare_fundamental_full_rebuild(
        ["000001.SZ"],
        canonical_scope_path=scope_path,
        canonical_market_pointer_path=market_pointer,
        canonical_membership_path=_membership_file(tmp_path),
        years=5,
        as_of="20240510",
        workers=1,
        pro=second_provider,
        checkpoint_root=checkpoint_root,
        max_attempts=1,
        requests_per_second=0,
        retry_backoff_seconds=0,
    )

    assert second_provider.calls == [("fina_indicator", "000001.SZ")]
    assert manifest["checkpoint"]["resumed_valid_request_count"] == 5
    assert manifest["checkpoint"]["requests_fetched_this_run"] == 1
    assert manifest["provider_calls_attempted"] == 7
    assert manifest["requests_retried"] == 1


def test_resume_refetches_only_malformed_key(tmp_path: Path) -> None:
    class _MalformedIncomeProvider(_Provider):
        def __getattr__(self, table: str):
            base = super().__getattr__(table)

            def fetch(**kwargs):
                frame = base(**kwargs)
                if table == "income" and not frame.empty:
                    frame["end_date"] = "not-a-date"
                return frame

            return fetch

    membership_path = _membership_file(tmp_path)
    checkpoint_root = tmp_path / "checkpoint"
    kwargs: dict[str, Any] = {
        "symbols": ["000001.SZ"],
        "canonical_scope_path": _scope_file(tmp_path),
        "canonical_market_pointer_path": _market_pointer_file(
            tmp_path,
            membership_path=membership_path,
        ),
        "canonical_membership_path": membership_path,
        "years": 5,
        "as_of": "20240510",
        "workers": 1,
        "checkpoint_root": checkpoint_root,
        "requests_per_second": 0,
        "retry_backoff_seconds": 0,
    }
    with pytest.raises(fundamental_mart.FundamentalFetchAuditError):
        fundamental_mart.fetch_tushare_fundamental_full_rebuild(
            pro=_MalformedIncomeProvider(),
            **kwargs,
        )

    second_provider = _Provider()
    _tables, manifest = fundamental_mart.fetch_tushare_fundamental_full_rebuild(
        pro=second_provider,
        **kwargs,
    )

    assert second_provider.calls == [("income", "000001.SZ")]
    assert manifest["checkpoint"]["resumed_valid_request_count"] == 5
    assert manifest["checkpoint"]["requests_fetched_this_run"] == 1


def test_resume_refetches_failed_financial_coverage_and_replaces_rows(
    tmp_path: Path,
) -> None:
    membership_path = _long_membership_file(tmp_path)
    checkpoint_root = tmp_path / "checkpoint"
    kwargs: dict[str, Any] = {
        "symbols": ["000001.SZ"],
        "canonical_scope_path": _scope_file(tmp_path),
        "canonical_market_pointer_path": _market_pointer_file(
            tmp_path,
            membership_path=membership_path,
        ),
        "canonical_membership_path": membership_path,
        "years": 5,
        "as_of": "20240510",
        "workers": 1,
        "checkpoint_root": checkpoint_root,
        "requests_per_second": 0,
        "retry_backoff_seconds": 0,
    }
    with pytest.raises(fundamental_mart.FundamentalFetchAuditError) as exc_info:
        fundamental_mart.fetch_tushare_fundamental_full_rebuild(
            pro=_CoverageProvider(income_complete=False, marker=1.0),
            **kwargs,
        )
    failed_income = next(
        outcome
        for outcome in exc_info.value.manifest["symbol_table_outcomes"]
        if outcome["table"] == "income"
    )
    assert failed_income["financial_coverage_passed"] is False
    assert failed_income["rows"] == 15

    second_provider = _CoverageProvider(marker=2.0)
    tables, manifest = fundamental_mart.fetch_tushare_fundamental_full_rebuild(
        pro=second_provider,
        **kwargs,
    )

    income = tables["income"]
    assert second_provider.calls == [("income", "000001.SZ")]
    assert manifest["checkpoint"]["resumed_valid_request_count"] == 5
    assert manifest["checkpoint"]["requests_fetched_this_run"] == 1
    assert len(income) == 16
    assert income["end_date"].nunique() == 16
    assert set(income["n_income"].tolist()) == {2.0}


def test_resume_refetches_incomplete_daily_history_and_replaces_rows(
    tmp_path: Path,
) -> None:
    membership_path = _long_membership_file(tmp_path)
    checkpoint_root = tmp_path / "checkpoint"
    kwargs: dict[str, Any] = {
        "symbols": ["000001.SZ"],
        "canonical_scope_path": _scope_file(tmp_path),
        "canonical_market_pointer_path": _market_pointer_file(
            tmp_path,
            membership_path=membership_path,
        ),
        "canonical_membership_path": membership_path,
        "years": 5,
        "as_of": "20240510",
        "workers": 1,
        "checkpoint_root": checkpoint_root,
        "requests_per_second": 0,
        "retry_backoff_seconds": 0,
    }
    with pytest.raises(fundamental_mart.FundamentalFetchAuditError) as exc_info:
        fundamental_mart.fetch_tushare_fundamental_full_rebuild(
            pro=_CoverageProvider(daily_complete=False, marker=1.0),
            **kwargs,
        )
    failed_daily = next(
        outcome
        for outcome in exc_info.value.manifest["symbol_table_outcomes"]
        if outcome["table"] == "daily_basic"
    )
    assert failed_daily["history_complete"] is False
    assert failed_daily["rows"] == 2

    second_provider = _CoverageProvider(marker=2.0)
    tables, manifest = fundamental_mart.fetch_tushare_fundamental_full_rebuild(
        pro=second_provider,
        **kwargs,
    )

    daily = tables["daily_basic"]
    assert second_provider.calls == [("daily_basic", "000001.SZ")]
    assert manifest["checkpoint"]["resumed_valid_request_count"] == 5
    assert manifest["checkpoint"]["requests_fetched_this_run"] == 1
    assert "20200112" not in set(daily["trade_date"].astype(str))
    assert daily["trade_date"].nunique() == len(daily)
    assert set(daily["total_mv"].tolist()) == {2.0}


def test_endpoint_audit_allows_forecast_empty_but_blocks_base_gap(tmp_path: Path) -> None:
    with pytest.raises(fundamental_mart.FundamentalFetchAuditError) as exc_info:
        fundamental_mart.fetch_tushare_fundamental_full_rebuild(
            ["000001.SZ"],
            canonical_scope_path=_scope_file(tmp_path),
            canonical_market_pointer_path=_market_pointer_file(tmp_path),
            canonical_membership_path=_membership_file(tmp_path),
            years=5,
            as_of="20240510",
            workers=1,
            pro=_Provider(daily_empty=True),
            checkpoint_root=tmp_path / "checkpoint",
            requests_per_second=0,
            retry_backoff_seconds=0,
        )

    audit = exc_info.value.manifest["endpoint_audit"]
    assert audit["endpoints"]["forecast"]["passed"] is True
    assert audit["endpoints"]["forecast"]["empty"] == 1
    assert audit["endpoints"]["daily_basic"]["request_denominator"] == 1
    assert "daily_basic_success_ratio_below_threshold" in audit["blockers"]


def test_endpoint_audit_rejects_nonempty_rows_missing_core_schema(
    tmp_path: Path,
) -> None:
    class _MissingCoreProvider(_Provider):
        def __getattr__(self, table: str):
            base = super().__getattr__(table)

            def fetch(**kwargs):
                frame = base(**kwargs)
                if table == "income" and not frame.empty:
                    return frame.drop(columns=["end_date"])
                return frame

            return fetch

    with pytest.raises(fundamental_mart.FundamentalFetchAuditError) as exc_info:
        fundamental_mart.fetch_tushare_fundamental_full_rebuild(
            ["000001.SZ"],
            canonical_scope_path=_scope_file(tmp_path),
            canonical_market_pointer_path=_market_pointer_file(tmp_path),
            canonical_membership_path=_membership_file(tmp_path),
            years=5,
            as_of="20240510",
            workers=1,
            pro=_MissingCoreProvider(),
            checkpoint_root=tmp_path / "checkpoint",
            requests_per_second=0,
            retry_backoff_seconds=0,
        )

    income = exc_info.value.manifest["endpoint_audit"]["endpoints"]["income"]
    assert income["malformed"] == 1
    assert income["passed"] is False


def test_scope_evidence_rejects_expired_membership_without_pointer_exception(
    tmp_path: Path,
) -> None:
    membership_path = tmp_path / "expired_membership.parquet"
    pd.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "list_date": "20200101",
                "effective_from": "20200101",
                "effective_to": "20240501",
                "delist_date": "20240501",
            }
        ]
    ).to_parquet(membership_path, index=False)

    with pytest.raises(ValueError, match="interval is not active as_of"):
        fundamental_mart.build_canonical_scope_evidence(
            ["000001.SZ"],
            canonical_path=_scope_file(tmp_path),
            market_pointer_path=_market_pointer_file(
                tmp_path,
                membership_path=membership_path,
            ),
            membership_path=membership_path,
            as_of="20240510",
        )


def test_scope_evidence_allows_expired_membership_only_with_bound_exception(
    tmp_path: Path,
) -> None:
    membership_path = tmp_path / "expired_membership.parquet"
    pd.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "list_date": "20200101",
                "effective_from": "20200101",
                "effective_to": "20240501",
                "delist_date": "20240501",
            }
        ]
    ).to_parquet(membership_path, index=False)

    evidence = fundamental_mart.build_canonical_scope_evidence(
        ["000001.SZ"],
        canonical_path=_scope_file(tmp_path),
        market_pointer_path=_market_pointer_file(
            tmp_path,
            non_blocking_absent_symbols=["000001.SZ"],
            membership_path=membership_path,
        ),
        membership_path=membership_path,
        as_of="20240510",
    )

    assert evidence["history_end_dates"] == {"000001.SZ": "20240501"}
    assert evidence["non_blocking_absent_symbols"] == ["000001.SZ"]


def test_scope_evidence_rejects_membership_end_before_listing(
    tmp_path: Path,
) -> None:
    membership_path = tmp_path / "invalid_order_membership.parquet"
    pd.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "list_date": "20240505",
                "effective_from": "20240505",
                "effective_to": "20240501",
                "delist_date": "20240501",
            }
        ]
    ).to_parquet(membership_path, index=False)

    with pytest.raises(ValueError, match="date order is invalid"):
        fundamental_mart.build_canonical_scope_evidence(
            ["000001.SZ"],
            canonical_path=_scope_file(tmp_path),
            market_pointer_path=_market_pointer_file(
                tmp_path,
                non_blocking_absent_symbols=["000001.SZ"],
                membership_path=membership_path,
            ),
            membership_path=membership_path,
            as_of="20240510",
        )


def test_scope_evidence_rejects_membership_not_bound_by_market_pointer(
    tmp_path: Path,
) -> None:
    canonical_membership = _membership_file(tmp_path)
    alternate_membership = tmp_path / "alternate_membership.parquet"
    pd.read_parquet(canonical_membership).to_parquet(
        alternate_membership,
        index=False,
    )

    with pytest.raises(ValueError, match="PIT membership binding mismatch"):
        fundamental_mart.build_canonical_scope_evidence(
            ["000001.SZ"],
            canonical_path=_scope_file(tmp_path),
            market_pointer_path=_market_pointer_file(
                tmp_path,
                membership_path=canonical_membership,
            ),
            membership_path=alternate_membership,
            as_of="20240510",
        )


def test_scope_evidence_rejects_nonempty_invalid_effective_to(
    tmp_path: Path,
) -> None:
    membership_path = tmp_path / "invalid_effective_to.parquet"
    pd.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "list_date": "20200101",
                "effective_from": "20200101",
                "effective_to": "not-a-date",
                "delist_date": "",
            }
        ]
    ).to_parquet(membership_path, index=False)

    with pytest.raises(ValueError, match="effective_to is invalid"):
        fundamental_mart.build_canonical_scope_evidence(
            ["000001.SZ"],
            canonical_path=_scope_file(tmp_path),
            market_pointer_path=_market_pointer_file(
                tmp_path,
                membership_path=membership_path,
            ),
            membership_path=membership_path,
            as_of="20240510",
        )


def test_scope_evidence_rejects_null_effective_to_as_open_interval(
    tmp_path: Path,
) -> None:
    membership_path = tmp_path / "null_effective_to.parquet"
    pd.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "list_date": "20200101",
                "effective_from": "20200101",
                "effective_to": pd.NA,
                "delist_date": "",
            }
        ]
    ).to_parquet(membership_path, index=False)

    with pytest.raises(ValueError, match="required date is null"):
        fundamental_mart.build_canonical_scope_evidence(
            ["000001.SZ"],
            canonical_path=_scope_file(tmp_path),
            market_pointer_path=_market_pointer_file(
                tmp_path,
                membership_path=membership_path,
            ),
            membership_path=membership_path,
            as_of="20240510",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("list_date", ""),
        ("effective_from", ""),
        ("effective_from", "20200101junk"),
        ("effective_to", "20240501junk"),
    ],
)
def test_scope_evidence_rejects_blank_or_suffixed_membership_dates(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    membership_path = tmp_path / f"invalid_{field}.parquet"
    row = {
        "symbol": "000001.SZ",
        "list_date": "20200101",
        "effective_from": "20200101",
        "effective_to": "",
        "delist_date": "",
    }
    row[field] = value
    pd.DataFrame([row]).to_parquet(membership_path, index=False)

    with pytest.raises(ValueError, match=f"{field} is invalid"):
        fundamental_mart.build_canonical_scope_evidence(
            ["000001.SZ"],
            canonical_path=_scope_file(tmp_path),
            market_pointer_path=_market_pointer_file(
                tmp_path,
                membership_path=membership_path,
            ),
            membership_path=membership_path,
            as_of="20240510",
        )


def test_endpoint_audit_blocks_unexcepted_daily_empty_below_ratio_tolerance() -> None:
    symbols = [f"{index:06d}.SZ" for index in range(1, 101)]
    outcomes = [
        _audit_outcome(
            symbol,
            table,
            (
                "empty"
                if symbol == symbols[-1] and table == "daily_basic"
                else "success"
            ),
            **(
                {"history_complete": True}
                if table == "daily_basic" and symbol != symbols[-1]
                else {}
            ),
        )
        for symbol in symbols
        for table in fundamental_mart.SOURCE_TABLES
    ]

    blocked = fundamental_mart._build_endpoint_audit(
        symbols,
        outcomes,
        policy=fundamental_mart.FundamentalEndpointAuditPolicy(),
    )
    excepted = fundamental_mart._build_endpoint_audit(
        symbols,
        outcomes,
        policy=fundamental_mart.FundamentalEndpointAuditPolicy(),
        daily_basic_empty_exception_symbols=[symbols[-1]],
    )
    incomplete_success_outcomes = [dict(outcome) for outcome in outcomes]
    incomplete_daily = next(
        outcome
        for outcome in incomplete_success_outcomes
        if outcome["symbol"] == symbols[-1] and outcome["table"] == "daily_basic"
    )
    incomplete_daily.update(
        status="success",
        rows_received=1,
        rows=1,
        history_complete=False,
    )
    incomplete_success = fundamental_mart._build_endpoint_audit(
        symbols,
        incomplete_success_outcomes,
        policy=fundamental_mart.FundamentalEndpointAuditPolicy(),
        daily_basic_empty_exception_symbols=[symbols[-1]],
    )

    assert blocked["passed"] is False
    assert blocked["daily_basic_history_incomplete_symbols"] == [symbols[-1]]
    assert "daily_basic_per_symbol_history_incomplete" in blocked["blockers"]
    assert excepted["passed"] is True
    assert excepted["daily_basic_history_exception_symbols"] == [symbols[-1]]
    assert incomplete_success["passed"] is False
    assert "daily_basic_per_symbol_history_incomplete" in incomplete_success[
        "blockers"
    ]
    assert "daily_basic_success_ratio_below_threshold" not in incomplete_success[
        "blockers"
    ]


def test_financial_coverage_blocks_latest_and_consecutive_baseline_gaps() -> None:
    symbol = "000001.SZ"
    baseline = fundamental_mart.matured_quarter_baseline(
        "20200101",
        "20200101",
        "20240510",
        "20240510",
    )
    tables = {
        table: pd.DataFrame(
            {
                "ts_code": symbol,
                "end_date": (
                    baseline[:-1]
                    if table == "income"
                    else baseline[:-2]
                    if table == "cashflow"
                    else baseline
                ),
            }
        )
        for table in fundamental_mart.FINANCIAL_SOURCE_TABLES
    }
    outcomes = [
        _audit_outcome(symbol, table, "success")
        for table in fundamental_mart.FINANCIAL_SOURCE_TABLES
    ]
    attached = fundamental_mart._attach_financial_coverage(
        [symbol],
        outcomes,
        tables,
        financial_start="20200101",
        as_of="20240510",
        scope_evidence={
            "listing_dates": {symbol: "20200101"},
            "history_end_dates": {symbol: "20240510"},
        },
        policy=fundamental_mart.FundamentalEndpointAuditPolicy(),
    )
    by_table = {str(item["table"]): item for item in attached}

    assert by_table["income"]["financial_coverage_passed"] is False
    assert "financial_latest_baseline_missing" in by_table["income"][
        "financial_coverage"
    ]["blockers"]
    assert by_table["cashflow"]["financial_coverage_passed"] is False
    assert "financial_consecutive_baseline_missing_above_threshold" in by_table[
        "cashflow"
    ]["financial_coverage"]["blockers"]
    assert by_table["fina_indicator"]["financial_coverage_passed"] is True


def test_financial_expected_zero_is_not_applicable_and_excluded_from_denominator() -> None:
    symbol = "000001.SZ"
    outcomes = [
        _audit_outcome(symbol, table, "empty")
        for table in fundamental_mart.FINANCIAL_SOURCE_TABLES
    ]
    outcomes.extend(
        [
            _audit_outcome(
                symbol,
                "daily_basic",
                "success",
                history_complete=True,
            ),
            _audit_outcome(symbol, "forecast", "empty"),
        ]
    )
    attached = fundamental_mart._attach_financial_coverage(
        [symbol],
        outcomes,
        {table: pd.DataFrame() for table in fundamental_mart.SOURCE_TABLES},
        financial_start="20240510",
        as_of="20240510",
        scope_evidence={
            "listing_dates": {symbol: "20240510"},
            "history_end_dates": {symbol: "20240510"},
        },
        policy=fundamental_mart.FundamentalEndpointAuditPolicy(),
    )
    audit = fundamental_mart._build_endpoint_audit(
        [symbol],
        attached,
        policy=fundamental_mart.FundamentalEndpointAuditPolicy(),
    )

    income = audit["endpoints"]["income"]
    assert income["financial_coverage_not_applicable"] == 1
    assert income["financial_coverage_denominator"] == 0
    assert income["financial_coverage_pass_ratio"] is None


def test_financial_prelisting_cross_table_periods_do_not_expand_denominator() -> None:
    symbol = "001220.SZ"
    outcomes = [
        _audit_outcome(symbol, table, "success")
        for table in fundamental_mart.FINANCIAL_SOURCE_TABLES
    ]
    tables = {
        "fina_indicator": pd.DataFrame(
            {"ts_code": symbol, "end_date": ["20191231", "20231231"]}
        ),
        "income": pd.DataFrame(
            {"ts_code": symbol, "end_date": ["20231231"]}
        ),
        "balancesheet": pd.DataFrame(
            {"ts_code": symbol, "end_date": ["20191231", "20231231"]}
        ),
        "cashflow": pd.DataFrame(
            {"ts_code": symbol, "end_date": ["20231231"]}
        ),
    }

    attached = fundamental_mart._attach_financial_coverage(
        [symbol],
        outcomes,
        tables,
        financial_start="20190714",
        as_of="20260714",
        scope_evidence={
            "listing_dates": {symbol: "20260203"},
            "history_end_dates": {symbol: "20260714"},
        },
        policy=fundamental_mart.FundamentalEndpointAuditPolicy(),
    )

    assert all(
        outcome["financial_coverage"]["status"] == "not_applicable"
        and outcome["financial_coverage_passed"] is True
        for outcome in attached
    )


def test_daily_history_boundary_tolerance_is_inclusive() -> None:
    expected_start = pd.Timestamp("2019-01-01")
    expected_end = pd.Timestamp("2024-01-01")
    inclusive = pd.bdate_range(expected_start + pd.Timedelta(days=62), expected_end)
    outside = pd.bdate_range(expected_start + pd.Timedelta(days=63), expected_end)

    inclusive_metrics = fundamental_mart._daily_history_coverage_metrics(
        pd.Series(inclusive),
        expected_start="20190101",
        expected_end="20240101",
        allow_tail_gap=False,
        boundary_tolerance_days=62,
    )
    outside_metrics = fundamental_mart._daily_history_coverage_metrics(
        pd.Series(outside),
        expected_start="20190101",
        expected_end="20240101",
        allow_tail_gap=False,
        boundary_tolerance_days=62,
    )

    assert inclusive_metrics["history_start_complete"] is True
    assert inclusive_metrics["history_complete"] is True
    assert outside_metrics["history_start_complete"] is False
    assert outside_metrics["history_complete"] is False


def test_scope_evidence_rejects_missing_canonical_bar_symbol(tmp_path: Path) -> None:
    membership_path = _membership_file(tmp_path)
    pointer_path = _market_pointer_file(
        tmp_path,
        membership_path=membership_path,
    )
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    bars_root = Path(pointer["table_root"])
    pd.DataFrame(
        [{"ts_code": "000002.SZ", "trade_date": "20240510"}]
    ).to_parquet(bars_root / "part.parquet", index=False)

    with pytest.raises(ValueError, match="bar bounds missing symbol"):
        fundamental_mart.build_canonical_scope_evidence(
            ["000001.SZ"],
            canonical_path=_scope_file(tmp_path),
            market_pointer_path=pointer_path,
            membership_path=membership_path,
            as_of="20240510",
            daily_start="20190510",
        )


def test_scope_evidence_rejects_symlinked_canonical_bar_partition(
    tmp_path: Path,
) -> None:
    membership_path = _membership_file(tmp_path)
    pointer_path = _market_pointer_file(
        tmp_path,
        membership_path=membership_path,
    )
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    bars_root = Path(pointer["table_root"])
    outside = tmp_path / "outside-bars"
    outside.mkdir()
    pd.DataFrame(
        [{"ts_code": "000001.SZ", "trade_date": "20240510"}]
    ).to_parquet(outside / "part.parquet", index=False)
    (bars_root / "linked-partition").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="bar dataset contains a symlink"):
        fundamental_mart.build_canonical_scope_evidence(
            ["000001.SZ"],
            canonical_path=_scope_file(tmp_path),
            market_pointer_path=pointer_path,
            membership_path=membership_path,
            as_of="20240510",
            daily_start="20190510",
        )


def test_scope_evidence_rejects_symlinked_canonical_bar_ancestor(
    tmp_path: Path,
) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    membership_path = _membership_file(real_root)
    pointer_path = _market_pointer_file(
        real_root,
        membership_path=membership_path,
    )
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    alias = tmp_path / "alias"
    alias.symlink_to(real_root, target_is_directory=True)
    pointer["table_root"] = str(alias / "canonical-bars")
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")

    with pytest.raises(ValueError, match="bar dataset contains a symlink"):
        fundamental_mart.build_canonical_scope_evidence(
            ["000001.SZ"],
            canonical_path=_scope_file(real_root),
            market_pointer_path=pointer_path,
            membership_path=membership_path,
            as_of="20240510",
            daily_start="20190510",
        )


def test_scope_evidence_binds_bar_hash_and_bounds_to_same_stable_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    membership_path = _membership_file(tmp_path)
    pointer_path = _market_pointer_file(
        tmp_path,
        membership_path=membership_path,
    )
    original_stable_read = fundamental_mart._stable_regular_file_bytes
    swapped = False

    def swap_after_first_bar_read(path: Path, *, label: str) -> bytes:
        nonlocal swapped
        payload = original_stable_read(path, label=label)
        if label == "canonical market bar dataset file" and not swapped:
            swapped = True
            bars = pd.read_parquet(path)
            bars.loc[0, "trade_date"] = "20240509"
            bars.to_parquet(path, index=False)
        return payload

    monkeypatch.setattr(
        fundamental_mart,
        "_stable_regular_file_bytes",
        swap_after_first_bar_read,
    )
    with pytest.raises(ValueError, match="bar dataset changed during read"):
        fundamental_mart.build_canonical_scope_evidence(
            ["000001.SZ"],
            canonical_path=_scope_file(tmp_path),
            market_pointer_path=pointer_path,
            membership_path=membership_path,
            as_of="20240510",
            daily_start="20190510",
        )
    assert swapped is True


def test_daily_empty_exception_requires_active_as_of_history_end() -> None:
    symbol = "000001.SZ"
    expired = fundamental_mart._active_daily_tail_gap_exceptions(
        {
            "non_blocking_absent_symbols": [symbol],
            "history_end_dates": {symbol: "20240501"},
        },
        as_of="20240510",
    )
    active = fundamental_mart._active_daily_tail_gap_exceptions(
        {
            "non_blocking_absent_symbols": [symbol],
            "history_end_dates": {symbol: "20240510"},
        },
        as_of="20240510",
    )

    assert expired == []
    assert active == [symbol]


def test_daily_history_audit_rejects_rows_clustered_into_early_years(
    tmp_path: Path,
) -> None:
    membership_path = tmp_path / "long_membership.parquet"
    pd.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "list_date": "20190510",
                "effective_from": "20190510",
                "effective_to": "",
                "delist_date": "",
            }
        ]
    ).to_parquet(membership_path, index=False)

    class _ClusteredHistoryProvider(_Provider):
        def __getattr__(self, table: str):
            if table != "daily_basic":
                return super().__getattr__(table)

            def fetch(**kwargs):
                symbol = kwargs["ts_code"]
                self.calls.append((table, symbol))
                clustered = pd.bdate_range("2019-05-10", "2021-05-10").append(
                    pd.DatetimeIndex([pd.Timestamp("2024-05-10")])
                )
                return pd.DataFrame(
                    {
                        "ts_code": symbol,
                        "trade_date": clustered.strftime("%Y%m%d"),
                        "total_mv": 1.0,
                        "circ_mv": 1.0,
                        "pe": 10.0,
                        "pb": 1.0,
                    }
                )

            return fetch

    with pytest.raises(fundamental_mart.FundamentalFetchAuditError) as exc_info:
        fundamental_mart.fetch_tushare_fundamental_full_rebuild(
            ["000001.SZ"],
            canonical_scope_path=_scope_file(tmp_path),
            canonical_market_pointer_path=_market_pointer_file(
                tmp_path,
                membership_path=membership_path,
            ),
            canonical_membership_path=membership_path,
            years=5,
            as_of="20240510",
            workers=1,
            pro=_ClusteredHistoryProvider(),
            checkpoint_root=tmp_path / "checkpoint",
            requests_per_second=0,
            retry_backoff_seconds=0,
        )

    outcome = next(
        item
        for item in exc_info.value.manifest["symbol_table_outcomes"]
        if item["table"] == "daily_basic"
    )
    assert outcome["observed_history_rows"] >= outcome["minimum_history_rows"]
    assert outcome["monthly_history_coverage_ratio"] < 0.90
    assert outcome["max_consecutive_missing_months"] > 2
    assert outcome["history_complete"] is False
    assert "daily_basic_per_symbol_history_incomplete" in exc_info.value.manifest[
        "endpoint_audit"
    ]["blockers"]
