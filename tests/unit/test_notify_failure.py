"""Owner alert for scheduled jobs: ledger line plus a best-effort local notification."""

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "notify_failure",
    Path(__file__).resolve().parents[2] / "scripts/operations/notify_failure.py",
)
notify_failure = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(notify_failure)


def test_alert_is_appended_to_the_ledger_without_a_popup(tmp_path, monkeypatch):
    monkeypatch.setenv("MYQUANT_ALERT_NOTIFICATION", "0")
    monkeypatch.setattr(notify_failure, "OSASCRIPT", "/nonexistent/osascript")

    for exit_code in (2, 3):
        notify_failure.notify(
            job="daily-factor-loop", exit_code=exit_code, detail='x "quoted"', workspace=tmp_path
        )

    lines = (tmp_path / "logs/alerts.jsonl").read_text().splitlines()
    assert [json.loads(line)["exit_code"] for line in lines] == [2, 3]
    assert json.loads(lines[0])["job"] == "daily-factor-loop"
    assert json.loads(lines[0])["detail"] == 'x "quoted"'


def test_missing_notifier_binary_and_unwritable_ledger_never_raise(tmp_path, monkeypatch):
    monkeypatch.delenv("MYQUANT_ALERT_NOTIFICATION", raising=False)
    monkeypatch.setattr(notify_failure, "OSASCRIPT", "/nonexistent/osascript")
    (tmp_path / "logs").write_text("not a directory")

    alert = notify_failure.notify(job="evening-close", exit_code=2, detail="", workspace=tmp_path)

    assert alert["job"] == "evening-close"


def test_launcher_detail_names_each_step_blocker_of_the_newest_attempt(tmp_path):
    older = tmp_path / "slot-2020-20260930T123555Z-1"
    newest = tmp_path / "slot-2020-20261001T122505Z-2"
    older.mkdir()
    newest.mkdir()
    (older / "maintenance.stdout.json").write_text('{"blockers":["OLD"]}\n')
    (newest / "recovery.stdout.json").write_text(
        '{"blockers":["core_handoff_recovery","calendar_freshness"],"status":"PARTIAL"}\n'
    )
    (newest / "veto-recovery.stdout.json").write_text(
        'not json\n{"blocker_code":"INTERNAL_ERROR","status":"ERROR"}\n'
    )
    older.touch()
    newest.touch()

    assert notify_failure.launcher_detail(tmp_path) == (
        "slot-2020-20261001T122505Z-2 recovery:core_handoff_recovery, "
        "recovery:calendar_freshness, veto-recovery:INTERNAL_ERROR"
    )
    assert notify_failure.launcher_detail(tmp_path / "missing") == ""
