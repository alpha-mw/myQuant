from __future__ import annotations

import copy

import pytest

from quant_investor.contracts import canonical_json_bytes, get_contract
from quant_investor.market.exchange_calendar_compilation import (
    build_exchange_calendar_compilation,
)
from quant_investor.system import SystemPreconditionError

BASE = "2026-08-16T00:00:00Z"


def _ref(label: str, kind: str) -> dict[str, str]:
    digit = format(sum(label.encode("utf-8")) % 16, "x")
    return {
        "kind": kind,
        "contract_sha256": get_contract(kind).contract_sha256,
        "artifact_id": label,
        "semantic_sha256": format((int(digit, 16) + 1) % 16, "x") * 64,
        "byte_sha256": format((int(digit, 16) + 2) % 16, "x") * 64,
    }


def _sources() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    issuers = {"BSE": "BSE_OFFICIAL", "SSE": "SSE_OFFICIAL", "SZSE": "SZSE_OFFICIAL"}
    for exchange in ("BSE", "SSE", "SZSE"):
        refs = sorted(
            [_ref(f"{exchange.lower()}-capture", "system.exchange_calendar_capture")],
            key=canonical_json_bytes,
        )
        admissions = sorted(
            [
                _ref(
                    f"{exchange.lower()}-admission",
                    "system.exchange_calendar_decoder_admission",
                )
            ],
            key=canonical_json_bytes,
        )
        indexes = sorted(
            [
                _ref(
                    f"{exchange.lower()}-index",
                    "system.exchange_calendar_index_closure",
                )
            ],
            key=canonical_json_bytes,
        )
        rows.append(
            {
                "exchange_id": exchange,
                "issuer": issuers[exchange],
                "weekly_rule_intervals": [
                    {
                        "start_date": "2024-01-01",
                        "end_date": "2026-08-14",
                        "weekdays": [1, 2, 3, 4, 5],
                    }
                ],
                "closure_dates": ["2024-01-01", "2025-01-01", "2026-01-01"],
                "session_intervals": [
                    {
                        "phase": "OPENING_CALL_AUCTION",
                        "opens_local": "09:15:00",
                        "closes_local": "09:25:00",
                    },
                    {
                        "phase": "MORNING_CONTINUOUS_AUCTION",
                        "opens_local": "09:30:00",
                        "closes_local": "11:30:00",
                    },
                    {
                        "phase": "AFTERNOON_CONTINUOUS_AUCTION",
                        "opens_local": "13:00:00",
                        "closes_local": "15:00:00",
                    },
                ],
                "native_capture_refs": refs,
                "decoder_admission_refs": admissions,
                "index_closure_refs": indexes,
            }
        )
    return rows


def _build(sources: list[dict[str, object]], *, cutoff: str = "2026-08-14") -> dict[str, object]:
    return build_exchange_calendar_compilation(
        compilation_id="official-cn-calendar-test",
        coverage_start_date="2024-01-01",
        cutoff_date=cutoff,
        release_ref=_ref("release", "system.release"),
        source_exchange_rows=sources,
        calendar_json_file_ref={"relative_path": "strict/calendar.json", "byte_sha256": "a" * 64},
        calendar_parquet_file_ref={
            "relative_path": "strict/exchange_calendar.parquet",
            "byte_sha256": "b" * 64,
        },
        contradiction_rows=[],
        created_at=BASE,
    )


def test_three_exchange_compilation_replays_exact_runtime_projection() -> None:
    with pytest.raises(SystemPreconditionError, match="shallow calendar compilation"):
        _build(_sources())


def test_one_exchange_date_drift_blocks_exchange_less_collapse() -> None:
    sources = _sources()
    sources[0]["closure_dates"] = sorted([*sources[0]["closure_dates"], "2024-01-02"])
    with pytest.raises(SystemPreconditionError, match="shallow calendar compilation"):
        _build(sources)


def test_rule_gap_and_short_coverage_fail_closed() -> None:
    sources = _sources()
    sources[0]["weekly_rule_intervals"] = [
        {"start_date": "2024-01-02", "end_date": "2026-08-14", "weekdays": [1, 2, 3, 4, 5]}
    ]
    with pytest.raises(SystemPreconditionError, match="shallow calendar compilation"):
        _build(sources)
    with pytest.raises(SystemPreconditionError, match="shallow calendar compilation"):
        _build(_sources(), cutoff="2024-06-30")


def test_contradiction_rows_never_mutate_the_calendar() -> None:
    sources = copy.deepcopy(_sources())
    with pytest.raises(SystemPreconditionError, match="shallow calendar compilation"):
        build_exchange_calendar_compilation(
            compilation_id="blocked",
            coverage_start_date="2024-01-01",
            cutoff_date="2026-08-14",
            release_ref=_ref("release", "system.release"),
            source_exchange_rows=sources,
            calendar_json_file_ref={
                "relative_path": "strict/calendar.json",
                "byte_sha256": "a" * 64,
            },
            calendar_parquet_file_ref={
                "relative_path": "strict/exchange_calendar.parquet",
                "byte_sha256": "b" * 64,
            },
            contradiction_rows=[{"date": "2024-01-01", "reason": "BAR_ON_CLOSED"}],
            created_at=BASE,
        )
