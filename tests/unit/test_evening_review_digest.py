"""Read-only evening review digest: reports what exists, MISSING for what does not."""

import importlib.util
import json
from pathlib import Path

import pandas as pd

_SPEC = importlib.util.spec_from_file_location(
    "evening_review_digest",
    Path(__file__).resolve().parents[2] / "scripts/operations/evening_review_digest.py",
)
review = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(review)


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _root(tmp_path, monkeypatch):
    maintenance = tmp_path / "maintenance"
    for name, path in {
        "MAINTENANCE": maintenance,
        "EVENING": tmp_path / "evening",
        "DASHBOARD": tmp_path / "dashboard.json",
        "JOURNAL": tmp_path / "journal",
        "POOL": tmp_path / "pool",
        "ALERTS": tmp_path / "alerts.jsonl",
    }.items():
        monkeypatch.setattr(review, name, path)
    return maintenance


def test_empty_workspace_reports_missing_without_inferring(tmp_path, monkeypatch):
    _root(tmp_path, monkeypatch)

    result = review.digest("2026-10-08")

    assert result["read_only"] is True
    assert result["launcher"] == {"status": "MISSING"}
    assert result["factor_state"] == {"status": "MISSING"}
    assert result["target_trade_date"] == "MISSING"
    assert result["dag"] == {"status": "MISSING"}
    assert result["top100"] == {"status": "MISSING"}
    assert result["evening_close"] == [] and result["alerts"] == []


def test_digest_follows_the_attempt_target_date(tmp_path, monkeypatch):
    maintenance = _root(tmp_path, monkeypatch)
    attempt = maintenance / "attempts/a/attempt.json"
    _write(
        attempt,
        {
            "target_date": "20260930",
            "status": "PARTIAL",
            "core_blockers": [],
            "blockers": ["MACRO_WRITE_VETO_ACTIVE"],
            "stage_results": [{"stage": "MARKET", "status": "READY"}],
        },
    )
    slot = maintenance / "launcher_attempts/slot-2020-20261002T122500Z-1"
    _write(slot / "ended.json", {"process_exit_code": 2})
    _write(slot / "maintenance.stdout.json", {"attempt_receipt_ref": {"path": str(attempt)}})
    _write(maintenance / "factor-loop-state.json", {"trade_date": "20260930", "phase": "X"})
    _write(
        tmp_path / "journal/20260930/dag-status.v1.json",
        {"status": "PARTIAL", "nodes": {"top100": {"command_status": "EXECUTED"}}},
    )
    _write(
        tmp_path / "evening/20261002/execute-211505.json",
        {"status": "NO_ACTION", "steps": [{"step": "calendar", "result": {"status": "NO_ACTION"}}]},
    )
    pool = tmp_path / "pool/2026-09-30"
    pool.mkdir(parents=True)
    pd.DataFrame({"rank": [2, 1], "symbol": ["B.SZ", "A.SH"]}).to_parquet(pool / "top100.parquet")
    (tmp_path / "alerts.jsonl").write_text(
        json.dumps({"at": "2026-10-02T20:46:14+08:00", "job": "x"})
        + "\n"
        + json.dumps({"at": "2026-10-01T20:46:14+08:00", "job": "y"})
        + "\n"
    )

    result = review.digest("2026-10-02", head=1)

    assert result["launcher"]["exit_code"] == 2
    assert result["launcher"]["attempt"]["stages"] == {"MARKET": "READY"}
    assert result["target_trade_date"] == "20260930"
    assert result["dag"]["nodes"] == {"top100": "EXECUTED"}
    assert result["evening_close"][0]["steps"] == {"calendar": "NO_ACTION"}
    assert result["top100"]["head"] == [{"rank": "1", "symbol": "A.SH"}]
    assert [row["job"] for row in result["alerts"]] == ["x"]
