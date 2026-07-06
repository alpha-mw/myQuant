from __future__ import annotations

import pandas as pd

from quant_investor.themes.replay import (
    PIT_INDUSTRY_LABEL_NOTE,
    build_theme_calibration_dataset,
    build_benchmark_forward_returns,
    compute_forward_metrics,
    extract_snapshot_theme_rows,
)


def _frame(
    closes: list[float],
    *,
    start: str = "2026-01-01",
    column: str = "trade_date",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            column: pd.date_range(start, periods=len(closes), freq="D"),
            "close": closes,
        }
    )


def test_compute_forward_metrics_basic():
    frame = _frame([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])

    metrics = compute_forward_metrics(
        frame=frame,
        as_of="20260101",
        horizons=(1, 3, 5),
    )

    assert metrics["data_available"] is True
    assert metrics["forward_return_5d"] == 0.5
    assert metrics["max_drawdown_10d"] == 0.0
    assert metrics["max_runup_10d"] == 0.6


def test_compute_forward_metrics_missing_as_of_safe():
    frame = pd.DataFrame({"close": [10.0, 11.0, 12.0]})

    metrics = compute_forward_metrics(frame=frame, as_of="20260101")

    assert metrics["data_available"] is False
    assert metrics["forward_return_5d"] is None
    assert metrics["unavailable_reason"]


def test_build_benchmark_forward_returns_median():
    frames = {
        "A": _frame([10.0, 10.0, 10.0, 10.0, 10.0, 11.0]),
        "B": _frame([10.0, 10.0, 10.0, 10.0, 10.0, 12.0]),
        "C": _frame([10.0, 10.0, 10.0, 10.0, 10.0, 13.0]),
    }

    benchmark = build_benchmark_forward_returns(
        frames=frames,
        as_of="20260101",
        horizons=(5,),
    )

    assert benchmark[5] == 0.2


def test_extract_snapshot_theme_rows_from_wrapper():
    snapshot = {
        "snapshot_schema_version": "theme_snapshot.v1",
        "market": "CN",
        "universe_key": "full_a",
        "as_of": "20260618",
        "theme_rotation": {
            "symbol_scores": {"000001.SZ": 0.7},
            "symbol_primary_theme": {"000001.SZ": "industry::banking"},
            "symbol_phase": {"000001.SZ": "confirmed_rotation"},
            "symbol_risk_flags": {"000001.SZ": ["theme_low_breadth"]},
            "theme_scores": {
                "industry::banking": {
                    "theme_name": "Banking",
                    "score": 70.0,
                    "confidence": 0.8,
                    "member_count": 12,
                }
            },
        },
    }

    rows = extract_snapshot_theme_rows(snapshot)

    assert rows == [
        {
            "symbol": "000001.SZ",
            "symbol_theme_score": 0.7,
            "primary_theme_id": "industry::banking",
            "primary_theme_name": "Banking",
            "phase": "confirmed_rotation",
            "risk_flags": ["theme_low_breadth"],
            "theme_score": 70.0,
            "theme_confidence": 0.8,
            "theme_member_count": 12,
        }
    ]


def test_extract_snapshot_theme_rows_malformed_safe():
    assert extract_snapshot_theme_rows({"theme_rotation": "bad"}) == []
    assert extract_snapshot_theme_rows({"symbol_scores": "bad"}) == []


def test_theme_calibration_dataset_marks_industry_labels_non_pit():
    snapshot = {
        "market": "CN",
        "universe_key": "full_a",
        "as_of": "20260101",
        "theme_rotation": {
            "market": "CN",
            "universe_key": "full_a",
            "as_of": "20260101",
            "symbol_scores": {"000001.SZ": 0.7},
            "symbol_primary_theme": {"000001.SZ": "industry::banking"},
            "symbol_phase": {"000001.SZ": "confirmed_rotation"},
            "symbol_risk_flags": {"000001.SZ": []},
            "theme_scores": {
                "industry::banking": {
                    "theme_name": "Banking",
                    "score": 70.0,
                    "confidence": 0.8,
                    "member_count": 12,
                }
            },
        },
    }

    dataset = build_theme_calibration_dataset(
        snapshots=[snapshot],
        frames={"000001.SZ": _frame([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])},
        horizons=(1, 3, 5),
        benchmark_horizons=(5,),
    )
    payload = dataset.to_dict()

    assert payload["metadata"]["pit_industry_labels"] is False
    assert payload["metadata"]["industry_label_note"] == PIT_INDUSTRY_LABEL_NOTE
    assert PIT_INDUSTRY_LABEL_NOTE in dataset.to_markdown()
