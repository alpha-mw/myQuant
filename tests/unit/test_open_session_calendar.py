"""Cover the first real producer of the Factor v4 open-session calendar.

`validate_open_session_calendar_v4` and `assess_candidate_maturity` have always
required this artifact, but every `latest_pointer_sha256` in the tree was a test
digest — no code cut one from real data. These tests pin the derivation rules
that the validator only checks after the fact.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from quant_investor.factors.governance_protocol_v4 import (
    validate_open_session_calendar_v4,
)
from quant_investor.factors.open_session_calendar import (
    OpenSessionCalendarError,
    build_open_session_calendar_v4,
    month_end_sessions,
    nonoverlapping_cohorts,
)


def _write_snapshot(root: Path, sessions: list[str], *, snapshot_id: str = "20260804T000000Z"):
    """Materialize a minimal strict-Parquet snapshot with the given sessions."""

    market = root / "parquet" / "cn"
    table = market / "_snapshots" / snapshot_id / "table" / "bars"
    table.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"ts_code": ["601989.SH"] * len(sessions), "trade_date": sessions, "close": [1.0] * len(sessions)}
    ).to_parquet(table / "part.parquet", index=False)

    manifest = market / "_snapshots" / f"{snapshot_id}.json"
    manifest.write_text(json.dumps({"snapshot_id": snapshot_id}), encoding="utf-8")
    (market / "_latest.json").write_text(
        json.dumps(
            {
                "snapshot_id": snapshot_id,
                "manifest_path": f"_snapshots/{snapshot_id}.json",
            }
        ),
        encoding="utf-8",
    )
    return market


def _weekday_sessions(count: int, start: str = "2026-01-05") -> list[str]:
    days = pd.bdate_range(start=start, periods=count)
    return [item.strftime("%Y%m%d") for item in days]


def test_calendar_is_accepted_by_the_v4_validator(tmp_path):
    _write_snapshot(tmp_path, _weekday_sessions(40))

    calendar = build_open_session_calendar_v4(data_root=tmp_path)

    validated = validate_open_session_calendar_v4(calendar.payload)
    assert validated["calendar_sha256"] == calendar.calendar_sha256
    assert calendar.snapshot_id == "20260804T000000Z"


def test_sessions_are_iso_sorted_and_distinct(tmp_path):
    sessions = _weekday_sessions(10)
    _write_snapshot(tmp_path, sessions + sessions)  # duplicated rows

    calendar = build_open_session_calendar_v4(data_root=tmp_path)

    dates = calendar.open_session_dates
    assert dates == sorted(set(dates))
    assert len(dates) == 10
    assert dates[0] == "2026-01-05"


def test_implausible_sessions_are_excluded_not_inherited(tmp_path):
    """The 19700101 rows still live in published snapshots."""

    _write_snapshot(tmp_path, ["19700101", *_weekday_sessions(5)])

    calendar = build_open_session_calendar_v4(data_root=tmp_path)

    assert calendar.excluded_sessions == ("1970-01-01",)
    assert "1970-01-01" not in calendar.open_session_dates
    assert len(calendar.open_session_dates) == 5


def test_weekend_session_fails_closed(tmp_path):
    _write_snapshot(tmp_path, [*_weekday_sessions(5), "20260110"])  # a Saturday

    with pytest.raises(OpenSessionCalendarError, match="weekend session"):
        build_open_session_calendar_v4(data_root=tmp_path)


def test_unparsable_trade_date_fails_closed(tmp_path):
    _write_snapshot(tmp_path, [*_weekday_sessions(3), "20261332"])

    with pytest.raises(OpenSessionCalendarError, match="not calendar dates"):
        build_open_session_calendar_v4(data_root=tmp_path)


def test_missing_pointer_fails_closed(tmp_path):
    with pytest.raises(OpenSessionCalendarError, match="missing strict Parquet pointer"):
        build_open_session_calendar_v4(data_root=tmp_path)


def test_calendar_binds_pointer_and_manifest_bytes(tmp_path):
    market = _write_snapshot(tmp_path, _weekday_sessions(5))
    first = build_open_session_calendar_v4(data_root=tmp_path)

    manifest = market / "_snapshots" / "20260804T000000Z.json"
    manifest.write_text(json.dumps({"snapshot_id": "20260804T000000Z", "note": "x"}), encoding="utf-8")
    second = build_open_session_calendar_v4(data_root=tmp_path)

    assert first.payload["manifest_sha256"] != second.payload["manifest_sha256"]
    assert first.calendar_sha256 != second.calendar_sha256


@pytest.mark.parametrize(
    "manifest_path",
    [
        "_snapshots/20260804T000000Z.json",  # market-relative
        "data/parquet/cn/_snapshots/20260804T000000Z.json",  # repo-relative, as published
    ],
)
def test_both_published_manifest_path_conventions_resolve(tmp_path, manifest_path):
    market = _write_snapshot(tmp_path, _weekday_sessions(5))
    pointer = market / "_latest.json"
    pointer.write_text(
        json.dumps({"snapshot_id": "20260804T000000Z", "manifest_path": manifest_path}),
        encoding="utf-8",
    )

    calendar = build_open_session_calendar_v4(data_root=tmp_path)

    assert len(calendar.open_session_dates) == 5


def test_manifest_path_is_not_resolved_by_basename_alone(tmp_path):
    """Stripping to `Path.name` would drop `_snapshots/` and hit the wrong file."""

    market = _write_snapshot(tmp_path, _weekday_sessions(5))
    decoy = market / "20260804T000000Z.json"
    decoy.write_text(json.dumps({"decoy": True}), encoding="utf-8")

    calendar = build_open_session_calendar_v4(data_root=tmp_path)

    real = market / "_snapshots" / "20260804T000000Z.json"
    import hashlib

    assert calendar.payload["manifest_sha256"] == hashlib.sha256(real.read_bytes()).hexdigest()


def test_month_end_sessions_use_actual_last_open_session(tmp_path):
    # 2026-01-30 is a Friday; 01-31 falls on a weekend and is not a session.
    _write_snapshot(tmp_path, ["20260129", "20260130", "20260202", "20260203"])

    calendar = build_open_session_calendar_v4(data_root=tmp_path)

    assert month_end_sessions(calendar) == ["2026-01-30", "2026-02-03"]


def test_cohorts_are_consecutive_non_overlapping_and_newest_aligned(tmp_path):
    _write_snapshot(tmp_path, _weekday_sessions(65))
    calendar = build_open_session_calendar_v4(data_root=tmp_path)

    cohorts = nonoverlapping_cohorts(calendar, size=30)

    assert len(cohorts) == 2  # 65 sessions -> 2 whole cohorts, 5 sessions dropped
    assert all(len(item["open_session_dates"]) == 30 for item in cohorts)
    # newest cohort ends on the newest session
    assert cohorts[-1]["end"] == calendar.open_session_dates[-1]
    # no overlap, and ascending
    assert cohorts[0]["end"] < cohorts[1]["start"]
    for cohort in cohorts:
        assert cohort["start"] == cohort["open_session_dates"][0]
        assert cohort["end"] == cohort["open_session_dates"][-1]
        assert cohort["horizon_days"] == 30
        assert cohort["calendar_sha256"] == calendar.calendar_sha256


def test_cohorts_are_empty_when_history_is_short(tmp_path):
    _write_snapshot(tmp_path, _weekday_sessions(29))
    calendar = build_open_session_calendar_v4(data_root=tmp_path)

    assert nonoverlapping_cohorts(calendar, size=30) == []
