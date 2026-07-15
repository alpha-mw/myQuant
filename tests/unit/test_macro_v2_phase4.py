from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from quant_investor.data._tushare_client import TushareClientPool
from quant_investor.macro import forward
from quant_investor.macro.forward import (
    MacroForwardError,
    forward_pointer_sha256,
    record_macro_forward_observation,
)
from quant_investor.macro.store import publish_observations

UTC = ZoneInfo("UTC")


def _row(period: str, available: str, value: float) -> dict[str, object]:
    return {
        "indicator_id": "cn.pmi_manufacturing",
        "dimension_type": "national",
        "industry_chain": "",
        "period_end": period,
        "release_at": available,
        "available_at": available,
        "vintage_id": f"observed:{period}",
        "value": value,
        "unit": "index",
        "frequency": "monthly",
        "source_system": "nbs_official",
        "source_record_id": f"pmi:{period}",
        "source_url": "https://www.stats.gov.cn/fixture",
        "fetched_at": available,
        "quality_status": "pass",
    }


def _observations(tmp_path: Path) -> Path:
    root = tmp_path / "observations"
    publish_observations(
        [
            _row("2024-01-31", "2024-02-01T01:30:00+00:00", 49.2),
            _row("2024-02-29", "2024-03-01T01:30:00+00:00", 49.1),
            _row("2024-03-31", "2024-04-01T01:30:00+00:00", 50.8),
        ],
        root=root,
        run_id="macro_g1",
    )
    return root


def _calendar(tmp_path: Path, *, end: str = "2024-05-15") -> Path:
    path = tmp_path / f"calendar_{end}.parquet"
    dates = pd.date_range("2024-05-09", end, freq="D")
    pd.DataFrame(
        {
            "cal_date": dates.strftime("%Y%m%d"),
            "is_open": [int(item.weekday() < 5) for item in dates],
        }
    ).to_parquet(path, index=False)
    return path


def _clock(monkeypatch, value: str) -> None:
    instant = datetime.fromisoformat(value).astimezone(UTC)
    monkeypatch.setattr(forward, "_utc_now", lambda: instant)


def _record(
    tmp_path: Path,
    monkeypatch,
    *,
    now: str = "2024-05-10T08:00:00+00:00",
    expected: str = "",
    calendar: Path | None = None,
    observations: Path | None = None,
):
    _clock(monkeypatch, now)
    return record_macro_forward_observation(
        observations_root=observations or _observations(tmp_path),
        calendar_path=calendar or _calendar(tmp_path),
        root=tmp_path / "forward",
        expected_pointer_sha256=expected,
    )


def test_first_completed_session_is_recorded_privately_and_not_eligible(
    tmp_path: Path, monkeypatch
):
    result = _record(tmp_path, monkeypatch)
    assert result["promoted"] is True
    assert result["latest_session"] == "2024-05-10"
    assert result["observed_forward_sessions"] == 1
    assert result["measurement_maturity_reached"] is False
    assert result["production_eligible"] is False
    assert (
        "outcome_stability_evidence_not_implemented"
        in result["maturity_blockers"]
    )

    market_root = tmp_path / "forward" / "CN"
    pointer = json.loads((market_root / "_latest.json").read_text())
    assert pointer["observer_only"] is True
    assert pointer["production_eligible"] is False
    assert pointer["applied"] is False
    generation = market_root / "_generations" / pointer["generation_id"]
    for path in generation.iterdir():
        assert os.stat(path).st_mode & 0o777 == 0o600
    assert os.stat(market_root / "_latest.json").st_mode & 0o777 == 0o600


def test_forward_pointer_rejects_invalid_observer_flags(
    tmp_path: Path,
    monkeypatch,
):
    observations = _observations(tmp_path)
    calendar = _calendar(tmp_path)
    _record(
        tmp_path,
        monkeypatch,
        observations=observations,
        calendar=calendar,
    )
    market_root = tmp_path / "forward" / "CN"
    pointer_path = market_root / "_latest.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["observer_only"] = False
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")

    with pytest.raises(
        MacroForwardError,
        match="macro_forward_pointer_observer_flags_invalid",
    ):
        _record(
            tmp_path,
            monkeypatch,
            expected=forward_pointer_sha256(tmp_path / "forward"),
            observations=observations,
            calendar=calendar,
        )


@pytest.mark.parametrize("generation_id", [".", ".."])
def test_forward_pointer_rejects_dot_generation_id(
    tmp_path: Path,
    monkeypatch,
    generation_id: str,
):
    _record(tmp_path, monkeypatch)
    pointer_path = tmp_path / "forward" / "CN" / "_latest.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["generation_id"] = generation_id
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")

    with pytest.raises(
        MacroForwardError,
        match="macro_forward_generation_id_invalid",
    ):
        _record(
            tmp_path,
            monkeypatch,
            expected=forward_pointer_sha256(tmp_path / "forward"),
        )


def test_same_session_is_idempotent_only_for_same_snapshot_and_generation(
    tmp_path: Path, monkeypatch
):
    observations = _observations(tmp_path)
    calendar = _calendar(tmp_path)
    first = _record(
        tmp_path,
        monkeypatch,
        observations=observations,
        calendar=calendar,
    )
    second = _record(
        tmp_path,
        monkeypatch,
        expected=first["pointer_sha256"],
        observations=observations,
        calendar=calendar,
    )
    assert second["promoted"] is False
    assert second["idempotent"] is True

    publish_observations(
        [_row("2024-04-30", "2024-05-11T01:30:00+00:00", 51.0)],
        root=observations,
        run_id="macro_g2",
    )
    with pytest.raises(MacroForwardError, match="same_session_drift"):
        _record(
            tmp_path,
            monkeypatch,
            expected=first["pointer_sha256"],
            observations=observations,
            calendar=calendar,
        )


def test_next_session_appends_but_skipping_a_session_is_blocked(
    tmp_path: Path, monkeypatch
):
    observations = _observations(tmp_path)
    calendar = _calendar(tmp_path)
    first = _record(
        tmp_path,
        monkeypatch,
        observations=observations,
        calendar=calendar,
    )
    second = _record(
        tmp_path,
        monkeypatch,
        now="2024-05-13T08:00:00+00:00",
        expected=first["pointer_sha256"],
        observations=observations,
        calendar=calendar,
    )
    assert second["observed_forward_sessions"] == 2
    with pytest.raises(MacroForwardError, match="session_gap_detected"):
        _record(
            tmp_path,
            monkeypatch,
            now="2024-05-15T08:00:00+00:00",
            expected=second["pointer_sha256"],
            observations=observations,
            calendar=calendar,
        )


def test_before_close_uses_previous_completed_session(
    tmp_path: Path, monkeypatch
):
    result = _record(
        tmp_path,
        monkeypatch,
        now="2024-05-10T06:59:00+00:00",
    )
    assert result["latest_session"] == "2024-05-09"


def test_stale_calendar_is_rejected_instead_of_backfilled(
    tmp_path: Path, monkeypatch
):
    with pytest.raises(MacroForwardError, match="calendar_stale"):
        _record(
            tmp_path,
            monkeypatch,
            now="2024-05-15T08:00:00+00:00",
            calendar=_calendar(tmp_path, end="2024-05-13"),
        )


def test_calendar_date_gap_is_rejected(tmp_path: Path, monkeypatch):
    path = _calendar(tmp_path)
    frame = pd.read_parquet(path)
    frame = frame.loc[frame["cal_date"] != "20240512"]
    frame.to_parquet(path, index=False)
    with pytest.raises(MacroForwardError, match="calendar_date_gap"):
        _record(tmp_path, monkeypatch, calendar=path)


def test_same_session_calendar_drift_is_rejected(tmp_path: Path, monkeypatch):
    observations = _observations(tmp_path)
    calendar = _calendar(tmp_path)
    first = _record(
        tmp_path,
        monkeypatch,
        observations=observations,
        calendar=calendar,
    )
    frame = pd.read_parquet(calendar)
    frame["source_revision"] = "changed"
    frame.to_parquet(calendar, index=False)
    with pytest.raises(MacroForwardError, match="same_session_drift"):
        _record(
            tmp_path,
            monkeypatch,
            expected=first["pointer_sha256"],
            observations=observations,
            calendar=calendar,
        )


def test_pointer_cas_and_none_are_rejected(tmp_path: Path, monkeypatch):
    observations = _observations(tmp_path)
    calendar = _calendar(tmp_path)
    first = _record(
        tmp_path,
        monkeypatch,
        observations=observations,
        calendar=calendar,
    )
    with pytest.raises(MacroForwardError, match="pointer_cas_mismatch"):
        _record(
            tmp_path,
            monkeypatch,
            now="2024-05-13T08:00:00+00:00",
            expected="0" * 64,
            observations=observations,
            calendar=calendar,
        )
    _clock(monkeypatch, "2024-05-13T08:00:00+00:00")
    with pytest.raises(MacroForwardError, match="expected_pointer_required"):
        record_macro_forward_observation(
            observations_root=observations,
            calendar_path=calendar,
            root=tmp_path / "forward",
            expected_pointer_sha256=None,
        )
    assert (
        forward_pointer_sha256(tmp_path / "forward") == first["pointer_sha256"]
    )


def test_tampered_ledger_fails_closed(tmp_path: Path, monkeypatch):
    observations = _observations(tmp_path)
    calendar = _calendar(tmp_path)
    first = _record(
        tmp_path,
        monkeypatch,
        observations=observations,
        calendar=calendar,
    )
    market_root = tmp_path / "forward" / "CN"
    pointer = json.loads((market_root / "_latest.json").read_text())
    ledger = (
        market_root
        / "_generations"
        / pointer["generation_id"]
        / "ledger.jsonl"
    )
    ledger.write_bytes(ledger.read_bytes() + b"{}\n")
    with pytest.raises(MacroForwardError, match="ledger_file_hash_mismatch"):
        _record(
            tmp_path,
            monkeypatch,
            now="2024-05-13T08:00:00+00:00",
            expected=first["pointer_sha256"],
            observations=observations,
            calendar=calendar,
        )


def test_insecure_artifact_mode_fails_closed(tmp_path: Path, monkeypatch):
    observations = _observations(tmp_path)
    calendar = _calendar(tmp_path)
    first = _record(
        tmp_path,
        monkeypatch,
        observations=observations,
        calendar=calendar,
    )
    market_root = tmp_path / "forward" / "CN"
    pointer = json.loads((market_root / "_latest.json").read_text())
    ledger = (
        market_root
        / "_generations"
        / pointer["generation_id"]
        / "ledger.jsonl"
    )
    os.chmod(ledger, 0o644)
    with pytest.raises(MacroForwardError, match="artifact_mode_unsafe"):
        _record(
            tmp_path,
            monkeypatch,
            now="2024-05-13T08:00:00+00:00",
            expected=first["pointer_sha256"],
            observations=observations,
            calendar=calendar,
        )


def test_directory_fsync_failure_leaves_pointer_absent(
    tmp_path: Path, monkeypatch
):
    observations = _observations(tmp_path)
    calendar = _calendar(tmp_path)
    original = forward._fsync_dir

    def fail_generations(path: Path) -> None:
        if path.name == "_generations":
            raise OSError("injected_forward_generations_fsync_failure")
        original(path)

    monkeypatch.setattr(forward, "_fsync_dir", fail_generations)
    _clock(monkeypatch, "2024-05-10T08:00:00+00:00")
    with pytest.raises(OSError, match="injected_forward"):
        record_macro_forward_observation(
            observations_root=observations,
            calendar_path=calendar,
            root=tmp_path / "forward",
            expected_pointer_sha256="",
        )
    market_root = tmp_path / "forward" / "CN"
    assert not (market_root / "_latest.json").exists()
    visible = [
        path
        for path in (market_root / "_generations").iterdir()
        if not path.name.startswith(".")
    ]
    assert visible == []


def test_pointer_directory_fsync_failure_keeps_referenced_generation(
    tmp_path: Path, monkeypatch
):
    observations = _observations(tmp_path)
    calendar = _calendar(tmp_path)
    original = forward._fsync_dir

    def fail_pointer_parent(path: Path) -> None:
        if path.name == "CN":
            raise OSError("injected_forward_pointer_fsync_failure")
        original(path)

    monkeypatch.setattr(forward, "_fsync_dir", fail_pointer_parent)
    _clock(monkeypatch, "2024-05-10T08:00:00+00:00")
    with pytest.raises(OSError, match="injected_forward_pointer"):
        record_macro_forward_observation(
            observations_root=observations,
            calendar_path=calendar,
            root=tmp_path / "forward",
            expected_pointer_sha256="",
        )
    pointer_hash = forward_pointer_sha256(tmp_path / "forward")
    assert pointer_hash

    monkeypatch.setattr(forward, "_fsync_dir", original)
    result = record_macro_forward_observation(
        observations_root=observations,
        calendar_path=calendar,
        root=tmp_path / "forward",
        expected_pointer_sha256=pointer_hash,
    )
    assert result["idempotent"] is True


def test_forward_recording_never_calls_tushare(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        TushareClientPool,
        "query",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("network forbidden")
        ),
    )
    assert _record(tmp_path, monkeypatch)["promoted"] is True
