#!/usr/bin/env python3
"""Tell the owner that a scheduled CN job did not finish cleanly.

launchd only appends to ``logs/*.log``, so a blocked or crashed night is silent.
This appends one JSON line to ``logs/alerts.jsonl`` and raises a local macOS
notification. It is best-effort and stdlib-only: it never raises, never changes
the caller's exit code, and makes no network, provider or broker call.

Set ``MYQUANT_ALERT_NOTIFICATION=0`` to keep the ledger line but skip the popup.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

WORKSPACE = Path("/Users/maxwell/mySpace/myQuant")
SHANGHAI = ZoneInfo("Asia/Shanghai")
OSASCRIPT = "/usr/bin/osascript"
# argv is passed to the script as data, so job output can never be run as AppleScript.
NOTIFICATION_SCRIPT = """on run argv
display notification (item 2 of argv) with title (item 1 of argv)
end run"""


def launcher_detail(launcher_attempts: Path) -> str:
    """Summarise the blocker codes the newest launcher attempt printed."""
    attempts = [path for path in launcher_attempts.glob("slot-*") if path.is_dir()]
    if not attempts:
        return ""
    newest = max(attempts, key=lambda path: path.stat().st_mtime)
    codes: list[str] = []
    for stdout in sorted(newest.glob("*.stdout.json")):
        for line in stdout.read_text(errors="replace").splitlines():
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if not isinstance(value, dict):
                continue
            found = [value.get("blocker_code"), *(value.get("blockers") or [])]
            step = stdout.name.removesuffix(".stdout.json")
            codes.extend(f"{step}:{code}" for code in found if isinstance(code, str) and code)
    return f"{newest.name} " + ", ".join(dict.fromkeys(codes))


def notify(*, job: str, exit_code: int, detail: str, workspace: Path = WORKSPACE) -> dict:
    """Append the alert to the ledger and raise the local notification."""
    alert = {
        "at": datetime.now(SHANGHAI).isoformat(timespec="seconds"),
        "job": job,
        "exit_code": exit_code,
        "detail": detail,
    }
    try:
        ledger = workspace / "logs/alerts.jsonl"
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(alert, ensure_ascii=False) + "\n")
    except OSError:
        pass
    if os.environ.get("MYQUANT_ALERT_NOTIFICATION") != "0":
        try:
            subprocess.run(
                [
                    OSASCRIPT,
                    "-e",
                    NOTIFICATION_SCRIPT,
                    f"myQuant {job} 未完成 (exit {exit_code})",
                    detail[:200] or "see logs/alerts.jsonl",
                ],
                capture_output=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            pass
    return alert


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    parser.add_argument("--exit-code", required=True, type=int)
    parser.add_argument("--detail", default="")
    parser.add_argument("--launcher-attempts", type=Path)
    parser.add_argument("--workspace-root", type=Path, default=WORKSPACE)
    args = parser.parse_args()
    detail = args.detail
    if not detail and args.launcher_attempts:
        try:
            detail = launcher_detail(args.launcher_attempts)
        except OSError:
            detail = ""
    notify(job=args.job, exit_code=args.exit_code, detail=detail, workspace=args.workspace_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
