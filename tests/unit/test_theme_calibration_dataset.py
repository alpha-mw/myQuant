from __future__ import annotations

import json

import pandas as pd

from quant_investor.themes.replay import (
    build_theme_calibration_dataset,
    build_theme_calibration_dataset_from_store,
)
from quant_investor.themes.storage import ThemeSnapshotStore


def _frame(closes: list[float], *, start: str = "2026-06-15") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.date_range(start, periods=len(closes), freq="D"),
            "close": closes,
        }
    )


def _snapshot() -> dict[str, object]:
    return {
        "snapshot_schema_version": "theme_snapshot.v1",
        "market": "CN",
        "universe_key": "full_a",
        "as_of": "20260618",
        "theme_rotation": {
            "schema_version": "theme_rotation.v1",
            "market": "CN",
            "universe_key": "full_a",
            "as_of": "20260618",
            "symbol_scores": {"000001.SZ": 0.7, "000002.SZ": 0.4},
            "symbol_primary_theme": {
                "000001.SZ": "industry::ai",
                "000002.SZ": "industry::banking",
            },
            "symbol_phase": {
                "000001.SZ": "confirmed_rotation",
                "000002.SZ": "distribution",
            },
            "symbol_risk_flags": {
                "000001.SZ": ["theme_low_breadth"],
                "000002.SZ": ["theme_distribution_risk"],
            },
            "theme_scores": {
                "industry::ai": {
                    "theme_name": "AI",
                    "score": 72.0,
                    "confidence": 0.8,
                    "member_count": 10,
                },
                "industry::banking": {
                    "theme_name": "Banking",
                    "score": 45.0,
                    "confidence": 0.5,
                    "member_count": 8,
                },
            },
        },
    }


def _frames() -> dict[str, pd.DataFrame]:
    return {
        "000001.SZ": _frame(
            [
                10.0,
                10.1,
                10.2,
                10.0,
                10.5,
                11.0,
                11.2,
                11.3,
                11.4,
                11.5,
                12.0,
                12.1,
                12.2,
                12.3,
                12.4,
                12.5,
                12.6,
                12.7,
                12.8,
                12.9,
                13.0,
                13.1,
                13.2,
                13.3,
            ]
        ),
        "000002.SZ": _frame(
            [
                20.0,
                20.0,
                20.0,
                20.0,
                19.8,
                19.6,
                19.4,
                19.2,
                19.0,
                18.8,
                18.6,
                18.5,
                18.4,
                18.3,
                18.2,
                18.1,
                18.0,
                17.9,
                17.8,
                17.7,
                17.6,
                17.5,
                17.4,
                17.3,
            ]
        ),
    }


def test_build_theme_calibration_dataset_from_dict_snapshot():
    dataset = build_theme_calibration_dataset(
        snapshots=[_snapshot()],
        frames=_frames(),
    )

    assert len(dataset.records) == 2
    first = dataset.records[0]
    assert first.forward_return_5d is not None
    assert first.forward_alpha_5d is not None
    assert isinstance(first.hit_5d, bool)
    assert dataset.metadata["record_count"] == 2


def test_build_theme_calibration_dataset_from_file_path(tmp_path):
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(_snapshot()), encoding="utf-8")

    dataset = build_theme_calibration_dataset(
        snapshots=[path],
        frames=_frames(),
    )

    assert len(dataset.records) == 2
    assert dataset.records[0].snapshot_path == str(path)


def test_theme_calibration_dataset_summary_by_phase_and_theme():
    dataset = build_theme_calibration_dataset(
        snapshots=[_snapshot()],
        frames=_frames(),
    )

    assert dataset.theme_summary["industry::ai"]["record_count"] == 1
    assert dataset.phase_summary["confirmed_rotation"]["record_count"] == 1
    assert dataset.phase_summary["distribution"]["record_count"] == 1


def test_theme_calibration_dataset_risk_flag_summary():
    dataset = build_theme_calibration_dataset(
        snapshots=[_snapshot()],
        frames=_frames(),
    )

    assert dataset.risk_flag_summary["theme_low_breadth"]["record_count"] == 1
    assert dataset.risk_flag_summary["theme_distribution_risk"]["record_count"] == 1


def test_theme_calibration_dataset_to_dataframe_and_markdown():
    dataset = build_theme_calibration_dataset(
        snapshots=[_snapshot()],
        frames=_frames(),
    )

    records = dataset.to_dataframe()
    summary = dataset.summary_dataframe()
    markdown = dataset.to_markdown()

    assert {"symbol", "forward_return_5d"} <= set(records.columns)
    assert not summary.empty
    assert "## Theme Replay Calibration Dataset" in markdown
    assert "Record count: 2" in markdown


def test_build_theme_calibration_dataset_missing_frame_safe():
    snapshot = _snapshot()
    snapshot["theme_rotation"]["symbol_scores"]["999999.SZ"] = 0.9

    dataset = build_theme_calibration_dataset(
        snapshots=[snapshot],
        frames=_frames(),
    )

    missing = [record for record in dataset.records if record.symbol == "999999.SZ"][0]
    assert missing.data_available is False
    assert dataset.metadata["missing_frame_count"] == 1


def test_build_theme_calibration_dataset_from_store(tmp_path):
    store = ThemeSnapshotStore(tmp_path)
    store.save(
        _snapshot()["theme_rotation"],
        market="CN",
        universe_key="full_a",
        as_of="20260618",
        run_id="run-a",
    )

    dataset = build_theme_calibration_dataset_from_store(
        store=store,
        frames=_frames(),
        market="CN",
        universe_key="full_a",
    )

    assert len(dataset.records) == 2
