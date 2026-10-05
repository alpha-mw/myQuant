#!/usr/bin/env python3
"""Evening preparation for the Paper account: evidence, intents, next plans.

Runs after the CN evening close for session `D` and does three mechanical things,
never a fill:

1. capture `D`'s exchange price limits (one provider request, sealed under
   `data/private/paper_evidence/`);
2. turn the plans that were written for `D` into `intent` + `eligibility` files,
   so the owner can review and run `paper risk-exit-run` per symbol;
3. write the plans for the next session from `D`'s sealed record.

Every step reports its own status; a step that cannot run (a record not yet
closed, no plans for `D`, missing evidence) is named in `blockers` and the others
still proceed. Nothing here writes to the Paper account.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

WORKSPACE = Path("/Users/maxwell/mySpace/myQuant")
SHADOW_ROOT = WORKSPACE / "data/private/paper_shadow"
RECORD_POINTER = (
    WORKSPACE
    / "results/strategy_records/CN/aggressive_tech_manufacturing/_record_store/current.v1.json"
)
ACCOUNT_ID = "aggressive-tech-manufacturing-paper-v1"
PYTHON = WORKSPACE / ".venv/bin/python"


def _run(script: str, *args: str) -> tuple[int, str]:
    completed = subprocess.run(
        [str(PYTHON), str(WORKSPACE / "scripts/operations" / script), *args],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
    )
    return completed.returncode, (completed.stdout + completed.stderr).strip()


def active_record_trade_date() -> str | None:
    try:
        pointer = json.loads(RECORD_POINTER.read_text())
    except (OSError, ValueError):
        return None
    record_dir = RECORD_POINTER.parent.parent / pointer["active_record_id"]
    summary = record_dir / "pnl_summary.csv"
    if not summary.exists():
        return None
    with summary.open() as handle:
        import csv

        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    return rows[-1]["quote_snapshot"].split("_")[0]


def plans_for(session: str) -> Path | None:
    """Newest plans.json whose eligible date is this session."""

    candidates: list[tuple[str, Path]] = []
    for path in SHADOW_ROOT.glob("*/*/plans.json"):
        try:
            value = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if value.get("eligible_from_trade_date") == session and value.get("orders"):
            candidates.append((path.parent.as_posix(), path))
    if not candidates:
        return None
    return sorted(candidates)[-1][1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trade-date", required=True, help="YYYYMMDD session")
    parser.add_argument("--plan-only", action="store_true", help="skip capture and intent emission")
    args = parser.parse_args()
    session = args.trade_date
    report: dict = {"trade_date": session, "account_id": ACCOUNT_ID, "steps": {}, "blockers": []}

    if not args.plan_only:
        code, output = _run("paper_capture_session_limits.py", "--trade-date", session, "--write")
        if code != 0:
            report["blockers"].append("LIMIT_CAPTURE_FAILED")
            report["steps"]["capture_limits"] = {"status": "BLOCKED", "output": output[-800:]}
        else:
            captured = json.loads(output)
            report["steps"]["capture_limits"] = {
                "status": "CAPTURED",
                "symbols": captured["symbol_count"],
                "evidence_path": captured["evidence_path"],
                "evidence_sha256": captured["evidence_sha256"],
            }

        plans = plans_for(session)
        if plans is None:
            report["blockers"].append("NO_PLANS_FOR_SESSION")
            report["steps"]["emit_intents"] = {"status": "NO_PLANS"}
        elif "capture_limits" not in report["steps"] or (
            report["steps"]["capture_limits"]["status"] != "CAPTURED"
        ):
            report["steps"]["emit_intents"] = {"status": "NO_EVIDENCE"}
        else:
            evidence = report["steps"]["capture_limits"]
            code, output = _run(
                "paper_session_eligibility.py",
                "--account-id",
                ACCOUNT_ID,
                "--trade-date",
                session,
                "--limits",
                evidence["evidence_path"],
                "--expected-limits-sha256",
                evidence["evidence_sha256"],
                "--plans",
                plans.relative_to(WORKSPACE).as_posix(),
                "--write",
            )
            if code != 0:
                report["blockers"].append("ELIGIBILITY_FAILED")
                report["steps"]["emit_intents"] = {
                    "status": "BLOCKED",
                    "output": output[-800:],
                }
            else:
                prepared = json.loads(output)
                report["steps"]["emit_intents"] = {
                    "status": "PREPARED",
                    "plans": plans.relative_to(WORKSPACE).as_posix(),
                    "prepared": prepared["prepared"],
                    "skipped": prepared["skipped"],
                    "pointer_sha256": prepared["pointer_sha256"],
                }

    record_day = active_record_trade_date()
    if record_day != session:
        report["blockers"].append(f"RECORD_NOT_CLOSED_FOR_SESSION:{record_day}")
        report["steps"]["plan_next_session"] = {"status": "SKIPPED", "record_day": record_day}
    else:
        code, output = _run("paper_shadow_orders.py", "--write")
        if code != 0:
            report["blockers"].append("PLAN_NEXT_SESSION_FAILED")
            report["steps"]["plan_next_session"] = {"status": "BLOCKED", "output": output[-800:]}
        else:
            receipt = json.loads(output[output.index("{") :])
            report["steps"]["plan_next_session"] = {
                "status": "PLANNED",
                "target_session": receipt["target_session"],
                "orders": [
                    {
                        "symbol": order["symbol"],
                        "action": order["action"],
                        "shares": order["shares"],
                    }
                    for order in receipt["orders"]
                ],
            }

    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
