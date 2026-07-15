from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from quant_investor.market import fundamental_mart


SYMBOL = "000001.SZ"


def _outcome(
    table: str,
    status: str,
    *,
    rows: int = 0,
    **extra: object,
) -> dict[str, object]:
    return {
        "schema_version": fundamental_mart.FUNDAMENTAL_REQUEST_OUTCOME_SCHEMA,
        "symbol": SYMBOL,
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


def _empty_tables() -> dict[str, pd.DataFrame]:
    return {
        table: pd.DataFrame() for table in fundamental_mart.SOURCE_TABLES
    }


class _CoverageProvider:
    def __init__(
        self,
        *,
        income_complete: bool,
        marker: float,
    ) -> None:
        self.income_complete = income_complete
        self.marker = marker
        self.calls: list[tuple[str, str]] = []

    def __getattr__(self, table: str):
        if table not in fundamental_mart.SOURCE_TABLES:
            raise AttributeError(table)

        def fetch(**kwargs: object) -> pd.DataFrame:
            symbol = str(kwargs["ts_code"])
            self.calls.append((table, symbol))
            if table == "forecast":
                return pd.DataFrame()
            if table == "daily_basic":
                dates = pd.bdate_range("2022-05-10", "2024-05-10")
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
            periods = pd.period_range("2019Q1", "2023Q4", freq="Q-DEC")
            end_dates = [period.end_time.normalize() for period in periods]
            if table == "income" and not self.income_complete:
                end_dates = end_dates[:-1]
            rows: list[dict[str, object]] = []
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
                rows.append(
                    {
                        "ts_code": symbol,
                        "ann_date": (
                            end_date + pd.Timedelta(days=60)
                        ).strftime("%Y%m%d"),
                        "end_date": end_date.strftime("%Y%m%d"),
                        **values,
                    }
                )
            return pd.DataFrame(rows)

        return fetch


def _scope_evidence() -> dict[str, object]:
    return {
        "listing_dates": {SYMBOL: "20200101"},
        "history_end_dates": {SYMBOL: "20240510"},
        "canonical_bar_first_dates": {SYMBOL: "20220510"},
        "canonical_bar_last_dates": {SYMBOL: "20240510"},
        "non_blocking_absent_symbols": [],
    }


@pytest.mark.parametrize(
    "outcome",
    [
        _outcome("income", "success", rows=1),
        _outcome(
            "income",
            "success",
            rows=1,
            financial_coverage={"status": "applicable", "passed": True},
            financial_coverage_passed=False,
        ),
    ],
)
def test_checkpoint_publish_rejects_missing_or_inconsistent_financial_coverage(
    tmp_path: Path,
    outcome: dict[str, object],
) -> None:
    checkpoint_root = fundamental_mart._safe_checkpoint_root(
        tmp_path / "checkpoint"
    )
    tables = _empty_tables()
    tables["income"] = pd.DataFrame(
        [{"ts_code": SYMBOL, "end_date": "20231231"}]
    )

    with pytest.raises(
        fundamental_mart.FundamentalFetchCheckpointError,
        match="financial coverage is missing or inconsistent",
    ):
        fundamental_mart._write_fetch_checkpoint(
            checkpoint_root,
            binding={"scope": "coverage-contract"},
            tables=tables,
            outcomes=[outcome],
            expected_pointer_sha256="",
            expected_revision=0,
        )

    assert not (checkpoint_root / "latest.json").exists()


def test_resume_never_treats_missing_inconsistent_or_empty_financial_as_clean() -> None:
    valid_not_applicable = {
        "financial_coverage": {
            "status": "not_applicable",
            "passed": True,
        },
        "financial_coverage_passed": True,
    }
    outcomes = [
        _outcome("income", "success", rows=1),
        _outcome(
            "income",
            "success",
            rows=1,
            financial_coverage={"status": "applicable", "passed": True},
            financial_coverage_passed=False,
        ),
        _outcome("income", "empty", **valid_not_applicable),
    ]

    assert all(
        fundamental_mart._checkpoint_outcome_requires_refetch(
            outcome,
            daily_basic_empty_exception_symbols=(),
        )
        for outcome in outcomes
    )


def test_batch_one_crash_resume_persists_coverage_and_replaces_failed_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _scope_evidence()
    monkeypatch.setattr(
        fundamental_mart,
        "_validate_canonical_scope_evidence",
        lambda _value, _symbols: evidence,
    )
    checkpoint_root = tmp_path / "checkpoint"
    original_write = fundamental_mart._write_fetch_checkpoint
    checkpoint_writes = 0

    def crash_after_first_checkpoint(*args: object, **kwargs: object):
        nonlocal checkpoint_writes
        snapshot = original_write(*args, **kwargs)
        checkpoint_writes += 1
        if checkpoint_writes == 1:
            raise RuntimeError("simulated crash after batch checkpoint")
        return snapshot

    monkeypatch.setattr(
        fundamental_mart,
        "_write_fetch_checkpoint",
        crash_after_first_checkpoint,
    )
    first_provider = _CoverageProvider(
        income_complete=False,
        marker=1.0,
    )
    fetch_kwargs: dict[str, Any] = {
        "symbols": [SYMBOL],
        "years": 2,
        "as_of": "20240510",
        "workers": 1,
        "canonical_scope_evidence": {"fixture": True},
        "checkpoint_root": checkpoint_root,
        "checkpoint_batch_size": 1,
        "requests_per_second": 0,
        "retry_backoff_seconds": 0,
        "symbol_pause_seconds": 0,
    }

    with pytest.raises(RuntimeError, match="simulated crash"):
        fundamental_mart._fetch_tushare_tables(
            pro=first_provider,
            **fetch_kwargs,
        )

    pointer = json.loads((checkpoint_root / "latest.json").read_text())
    generation_root = (checkpoint_root / pointer["manifest_path"]).parent
    persisted = json.loads(
        (generation_root / "request_outcomes.json").read_text()
    )["outcomes"]
    failed_income = next(
        outcome for outcome in persisted if outcome["table"] == "income"
    )
    assert failed_income["financial_coverage_passed"] is False
    assert failed_income["financial_coverage"]["status"] == "applicable"

    replacement_provider = _CoverageProvider(
        income_complete=True,
        marker=2.0,
    )
    tables, manifest = fundamental_mart._fetch_tushare_tables(
        pro=replacement_provider,
        **fetch_kwargs,
    )

    assert replacement_provider.calls == [("income", SYMBOL)]
    assert manifest["checkpoint"]["resumed_valid_request_count"] == 5
    assert manifest["checkpoint"]["requests_fetched_this_run"] == 1
    assert len(tables["income"]) == 20
    assert set(tables["income"]["n_income"].tolist()) == {2.0}

    clean_provider = _CoverageProvider(income_complete=True, marker=3.0)
    _tables, clean_manifest = fundamental_mart._fetch_tushare_tables(
        pro=clean_provider,
        **fetch_kwargs,
    )
    assert clean_provider.calls == []
    assert clean_manifest["checkpoint"]["requests_fetched_this_run"] == 0


def test_candidate_tamper_during_pointer_cas_is_caught_before_switch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_root = fundamental_mart._safe_checkpoint_root(
        tmp_path / "checkpoint"
    )
    binding = {"scope": "post-cas-recheck"}
    tables = _empty_tables()
    first = fundamental_mart._write_fetch_checkpoint(
        checkpoint_root,
        binding=binding,
        tables=tables,
        outcomes=[],
        expected_pointer_sha256="",
        expected_revision=0,
    )
    pointer_before = (checkpoint_root / "latest.json").read_bytes()
    original_read = fundamental_mart._stable_regular_file_bytes
    tampered = False

    def tamper_after_pointer_cas(path: Path, *, label: str) -> bytes:
        nonlocal tampered
        payload = original_read(path, label=label)
        if label == "checkpoint pointer CAS readback" and not tampered:
            tampered = True
            candidates = sorted(
                (checkpoint_root / "_generations").glob(
                    "checkpoint_00000002_*"
                )
            )
            assert len(candidates) == 1
            outcomes_path = candidates[0] / "request_outcomes.json"
            outcomes_path.write_bytes(outcomes_path.read_bytes() + b" ")
        return payload

    monkeypatch.setattr(
        fundamental_mart,
        "_stable_regular_file_bytes",
        tamper_after_pointer_cas,
    )

    with pytest.raises(
        fundamental_mart.FundamentalFetchCheckpointError,
        match="candidate outcomes changed before publication",
    ):
        fundamental_mart._write_fetch_checkpoint(
            checkpoint_root,
            binding=binding,
            tables=tables,
            outcomes=[],
            expected_pointer_sha256=first.pointer_sha256,
            expected_revision=first.revision,
        )

    assert tampered is True
    assert (checkpoint_root / "latest.json").read_bytes() == pointer_before
