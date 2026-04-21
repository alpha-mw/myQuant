"""Tests for CN freshness mode: stable vs strict trade date resolution."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from quant_investor.market.cn_symbol_status import (
    CNSymbolLocalStatusResult,
    evaluate_batch_completeness,
    evaluate_symbol_local_status,
)


def _make_csv_reader(frames: dict[str, pd.DataFrame | None]):
    """Build a fake SharedCSVReader that serves pre-built frames."""
    reader = MagicMock()

    def _resolve(symbol, **_kw):
        return f"/fake/{symbol}.csv" if symbol in frames else None

    def _read(symbol, **_kw):
        frame = frames.get(symbol)
        if frame is None:
            frame = pd.DataFrame()
        return SimpleNamespace(frame=frame, issues=[])

    reader.resolve_symbol_path = _resolve
    reader.read_symbol_frame = _read
    return reader


def _frame_with_date(date_str: str) -> pd.DataFrame:
    return pd.DataFrame({"trade_date": [date_str], "close": [10.0]})


class TestFreshnessFieldPropagation:
    def test_result_carries_three_date_model(self):
        reader = _make_csv_reader({"000001.SZ": _frame_with_date("20260403")})
        result = evaluate_symbol_local_status(
            "000001.SZ",
            category="full_a",
            resolver=MagicMock(),
            csv_reader=reader,
            latest_trade_date="20260403",
            allowed_stale_symbols=None,
            suspended_symbols=None,
            freshness_mode="stable",
            strict_trade_date="20260403",
            stable_trade_date="20260402",
        )
        assert result.freshness_mode == "stable"
        assert result.strict_trade_date == "20260403"
        assert result.stable_trade_date == "20260402"
        assert result.effective_target_trade_date == "20260403"

    def test_stable_mode_uses_t_minus_1(self):
        reader = _make_csv_reader({"000001.SZ": _frame_with_date("20260402")})
        result = evaluate_symbol_local_status(
            "000001.SZ",
            category="full_a",
            resolver=MagicMock(),
            csv_reader=reader,
            latest_trade_date="20260402",
            allowed_stale_symbols=None,
            suspended_symbols=None,
            freshness_mode="stable",
            strict_trade_date="20260403",
            stable_trade_date="20260402",
        )
        assert result.local_status == "up_to_date"
        assert result.is_complete is True

    def test_strict_mode_requires_t0(self):
        reader = _make_csv_reader({"000001.SZ": _frame_with_date("20260402")})
        result = evaluate_symbol_local_status(
            "000001.SZ",
            category="full_a",
            resolver=MagicMock(),
            csv_reader=reader,
            latest_trade_date="20260403",
            allowed_stale_symbols=None,
            suspended_symbols=None,
            freshness_mode="strict",
            strict_trade_date="20260403",
            stable_trade_date="20260402",
        )
        assert result.local_status == "stale"
        assert result.is_blocking is True

    def test_missing_symbol_carries_freshness_metadata(self):
        reader = _make_csv_reader({})
        result = evaluate_symbol_local_status(
            "999999.SZ",
            category="full_a",
            resolver=MagicMock(),
            csv_reader=reader,
            latest_trade_date="20260403",
            allowed_stale_symbols=None,
            suspended_symbols=None,
            freshness_mode="stable",
            strict_trade_date="20260403",
            stable_trade_date="20260402",
        )
        assert result.local_status == "missing"
        assert result.freshness_mode == "stable"
        assert result.effective_target_trade_date == "20260403"


class TestBatchCompleteness:
    def test_batch_evaluates_multiple_symbols(self):
        reader = _make_csv_reader({
            "000001.SZ": _frame_with_date("20260403"),
            "600519.SH": _frame_with_date("20260402"),
        })
        results = evaluate_batch_completeness(
            ["000001.SZ", "600519.SH", "999999.SZ"],
            category="full_a",
            resolver=MagicMock(),
            csv_reader=reader,
            latest_trade_date="20260403",
            freshness_mode="stable",
            strict_trade_date="20260403",
            stable_trade_date="20260402",
        )
        assert len(results) == 3
        assert results["000001.SZ"].local_status == "up_to_date"
        assert results["600519.SH"].local_status == "stale"
        assert results["999999.SZ"].local_status == "missing"

    def test_batch_skips_empty_symbols(self):
        reader = _make_csv_reader({})
        results = evaluate_batch_completeness(
            ["", "  ", "000001.SZ"],
            category="full_a",
            resolver=MagicMock(),
            csv_reader=reader,
            latest_trade_date="20260403",
        )
        assert len(results) == 1
        assert "000001.SZ" in results

    def test_batch_suspended_stale(self):
        reader = _make_csv_reader({
            "000001.SZ": _frame_with_date("20260401"),
        })
        results = evaluate_batch_completeness(
            ["000001.SZ"],
            category="full_a",
            resolver=MagicMock(),
            csv_reader=reader,
            latest_trade_date="20260403",
            suspended_symbols=["000001.SZ"],
        )
        assert results["000001.SZ"].local_status == "suspended_stale"
        assert results["000001.SZ"].is_complete is True
