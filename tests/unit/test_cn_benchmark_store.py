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
