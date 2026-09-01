from __future__ import annotations

from pathlib import Path

import pytest

from quant_investor.market.cn_benchmark_store import (
    CNBenchmarkCASMismatch,
    EMPTY_POINTER_SHA256,
    REQUIRED_CODES,
    load_generation,
    publish_generation,
)
from scripts.operations import run_cn_benchmark_close as producer


def _rows() -> list[dict[str, object]]:
    return [
        {
            "date": day,
            "ts_code": code,
            "close": 1000.0 + index,
            "source_system": "fixture.index_daily",
            "coverage": "exact_close",
            "value_date": day,
        }
        for day in ("2026-08-24", "2026-08-25")
        for index, code in enumerate(REQUIRED_CODES)
    ]


def test_publish_and_load_immutable_benchmark_generation(tmp_path: Path) -> None:
    result = publish_generation(
        tmp_path,
        rows=_rows(),
        generation_id="benchmark-20260825-test",
        captured_at="2026-08-25T10:00:00Z",
        expected_pointer_sha256=EMPTY_POINTER_SHA256,
        acquisition_receipt_ref={"path": "private/capture.json", "sha256": "a" * 64},
    )
    loaded = load_generation(tmp_path)

    assert loaded == result
    assert loaded["pointer"]["end_date"] == "2026-08-25"
    assert len(loaded["rows"]) == 6
    assert loaded["pointer"]["broker_order_trade_authority"] is False


def test_benchmark_generation_requires_all_three_exact_rows(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="complete three-index day"):
        publish_generation(
            tmp_path,
            rows=_rows()[:-1],
            generation_id="benchmark-incomplete-test",
            captured_at="2026-08-25T10:00:00Z",
            expected_pointer_sha256=EMPTY_POINTER_SHA256,
            acquisition_receipt_ref={"path": "private/capture.json", "sha256": "a" * 64},
        )


def test_benchmark_pointer_cas_conflict(tmp_path: Path) -> None:
    published = publish_generation(
        tmp_path,
        rows=_rows(),
        generation_id="benchmark-first-test",
        captured_at="2026-08-25T10:00:00Z",
        expected_pointer_sha256=EMPTY_POINTER_SHA256,
        acquisition_receipt_ref={"path": "private/capture.json", "sha256": "a" * 64},
    )
    assert published["pointer_sha256"] != EMPTY_POINTER_SHA256
    with pytest.raises(CNBenchmarkCASMismatch):
        publish_generation(
            tmp_path,
            rows=_rows(),
            generation_id="benchmark-second-test",
            captured_at="2026-08-25T10:01:00Z",
            expected_pointer_sha256=EMPTY_POINTER_SHA256,
            acquisition_receipt_ref={"path": "private/capture-2.json", "sha256": "b" * 64},
        )


def test_tushare_capture_uses_monthly_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas as pd

    calls: list[tuple[str, str, str]] = []

    class FakePro:
        def index_daily(self, *, ts_code: str, start_date: str, end_date: str):
            calls.append((ts_code, start_date, end_date))
            return pd.DataFrame([{"ts_code": ts_code, "trade_date": start_date, "close": 1000.0}])

    monkeypatch.setattr(producer, "create_tushare_pro", lambda *_args: FakePro())
    monkeypatch.setattr(producer, "TUSHARE_REQUEST_INTERVAL_SECONDS", 0.0)

    rows = producer._provider_rows(
        "token-value-is-never-recorded",
        start_date="2026-03-17",
        end_date="2026-08-31",
        source="tushare",
    )

    assert len(calls) == 18
    assert calls[0][1:] == ("20260317", "20260331")
    assert calls[-1][1:] == ("20260801", "20260831")
    assert len(rows) == 18
