from pathlib import Path

import pytest

from scripts import build_v17_v3_current_shadow as subject


def _calendar_row(query_id: str, path: str) -> dict[str, str]:
    return {
        "expected_sha256": "a" * 64,
        "kind": "parquet",
        "path": path,
        "query_id": query_id,
    }


def test_calendar_lookup_accepts_one_date_versioned_trade_calendar() -> None:
    inventory = {
        "acquisition": {
            "files": [
                _calendar_row(
                    "trade_cal_cn_2016_20260727",
                    "/private/calendar.parquet",
                )
            ]
        }
    }

    assert subject._find_calendar_path(inventory) == (
        Path("/private/calendar.parquet"),
        "a" * 64,
    )


def test_calendar_lookup_rejects_ambiguous_trade_calendars() -> None:
    inventory = {
        "acquisition": {
            "files": [
                _calendar_row("trade_cal_cn_2016_20260724", "/private/old.parquet"),
                _calendar_row("trade_cal_cn_2016_20260727", "/private/new.parquet"),
            ]
        }
    }

    with pytest.raises(
        subject.CurrentShadowBuildError,
        match="source_dataset_calendar_missing",
    ):
        subject._find_calendar_path(inventory)
