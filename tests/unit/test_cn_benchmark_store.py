from __future__ import annotations

from pathlib import Path

import pytest

from quant_investor.market.cn_benchmark_store import (
    CNBenchmarkCASMismatch,
    CNBenchmarkStoreError,
    EMPTY_POINTER_SHA256,
    REQUIRED_CODES,
    compatibility_csv_bytes,
    load_generation,
    load_immutable_generation,
    publish_generation,
)
from quant_investor.market.cn_benchmark_capture import request_partition


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


def test_explicit_immutable_generation_reproduces_legacy_alias_after_latest_advances(
    tmp_path: Path,
) -> None:
    first = publish_generation(
        tmp_path,
        rows=_rows(),
        generation_id="benchmark-first-test",
        captured_at="2026-08-25T10:00:00Z",
        expected_pointer_sha256=EMPTY_POINTER_SHA256,
        acquisition_receipt_ref={"path": "private/first.json", "sha256": "a" * 64},
    )
    old_alias = compatibility_csv_bytes(first["rows"])
    extended = [
        *_rows(),
        *[
            {
                "date": "2026-08-26",
                "ts_code": code,
                "close": 1100.0 + index,
                "source_system": "fixture.index_daily",
                "coverage": "exact_close",
                "value_date": "2026-08-26",
            }
            for index, code in enumerate(REQUIRED_CODES)
        ],
    ]
    publish_generation(
        tmp_path,
        rows=extended,
        generation_id="benchmark-second-test",
        captured_at="2026-08-26T10:00:00Z",
        expected_pointer_sha256=first["pointer_sha256"],
        acquisition_receipt_ref={"path": "private/second.json", "sha256": "b" * 64},
    )

    historical = load_immutable_generation(tmp_path, "benchmark-first-test")
    assert compatibility_csv_bytes(historical["rows"]) == old_alias
    assert compatibility_csv_bytes(load_generation(tmp_path)["rows"]) != old_alias


def test_explicit_immutable_generation_rejects_missing_or_tampered_bytes(tmp_path: Path) -> None:
    publish_generation(
        tmp_path,
        rows=_rows(),
        generation_id="benchmark-first-test",
        captured_at="2026-08-25T10:00:00Z",
        expected_pointer_sha256=EMPTY_POINTER_SHA256,
        acquisition_receipt_ref={"path": "private/first.json", "sha256": "a" * 64},
    )
    with pytest.raises(CNBenchmarkStoreError, match="regular file"):
        load_immutable_generation(tmp_path, "benchmark-missing-test")
    series = tmp_path / "_generations/benchmark-first-test/series.parquet"
    series.write_bytes(series.read_bytes() + b"tamper")
    with pytest.raises(CNBenchmarkStoreError, match="series closure mismatch"):
        load_immutable_generation(tmp_path, "benchmark-first-test")


def test_tushare_capture_uses_monthly_chunks() -> None:
    partitions = request_partition("2026-03-17", "2026-08-31")
    assert len(partitions) == 18
    assert partitions[0] == {
        "ts_code": REQUIRED_CODES[0],
        "start_date": "20260317",
        "end_date": "20260331",
    }
    assert partitions[-1] == {
        "ts_code": REQUIRED_CODES[-1],
        "start_date": "20260801",
        "end_date": "20260831",
    }
