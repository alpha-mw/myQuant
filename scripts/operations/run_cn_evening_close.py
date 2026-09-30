#!/usr/bin/env python3
"""Deterministic CN evening close: benchmark -> event closure -> official close -> Dashboard.

Runs after the launchd 20:25 daily maintenance. Every step is an existing,
SHA-bound command; this file only sequences them, pins the exact preimage SHAs
each one needs, and stops at the first failure (fail-closed). Default is a plan
with no writes; ``--execute`` performs them. A receipt of every step is written
to ``data/private/cn_evening_close/<YYYYMMDD>/``.

Official-close commands run from the frozen release checkout/install that the
daily maintenance uses; the Dashboard exporter runs from the workspace, like the
Hermes Dashboard job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

WORKSPACE = Path("/Users/maxwell/mySpace/myQuant")
RELEASE_COMMIT = "7cdb1f265f5c4b6a2ae8e799012a963cf2904b6c"
RELEASE_CHECKOUT = Path(
    f"/Users/maxwell/mySpace/myQuant-release-checkouts/{RELEASE_COMMIT}-unified-runtime"
)
RELEASE_PYTHON = Path(
    "/Users/maxwell/mySpace/myQuant-release-authority/"
    f"{RELEASE_COMMIT}-unified-runtime/installs/"
    f"{RELEASE_COMMIT}-1ef1b686e5e4e3d09fed188ec9386267fe3d57b234d7b3a08148957bf142bc0b/bin/python"
)
WORKSPACE_PYTHON = WORKSPACE / ".venv/bin/python"
RECORD_ROOT = WORKSPACE / "results/strategy_records/CN/aggressive_tech_manufacturing"
MAINTENANCE_ROOT = WORKSPACE / "data/private/cn_daily_maintenance"
CLOSE_POLICY = "operations/policies/cn-daily-official-close-policy.v1.json"
NO_EVENT_POLICY = "operations/policies/cn-daily-no-event-default.v1.json"
TRADE_INBOX = WORKSPACE / "operations/owner_trades/CN"
RECEIPT_ROOT = WORKSPACE / "data/private/cn_evening_close"
SHANGHAI = ZoneInfo("Asia/Shanghai")


class Blocked(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_json(argv: list[str], *, cwd: Path, label: str, timeout: int = 1800) -> dict:
    completed = subprocess.run(
        argv, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip().startswith("{")]
    if not lines:
        raise Blocked(
            f"{label}: no JSON output (exit {completed.returncode}): "
            f"{completed.stderr.strip()[-400:]}"
        )
    value = json.loads(lines[-1])
    if completed.returncode != 0 or value.get("ok") is False:
        raise Blocked(f"{label}: {value.get('error') or value.get('blocker') or value}")
    return value


def maintenance_attempt(day: str) -> tuple[Path, str, Path, str]:
    """Return today's 2020-slot attempt receipt and its close-session receipt."""
    compact = day.replace("-", "")
    candidates = []
    for slot in sorted((MAINTENANCE_ROOT / "launcher_attempts").glob(f"slot-2020-{compact}T*")):
        stdout = slot / "maintenance.stdout.json"
        if not (slot / "ended.json").is_file() or not stdout.is_file():
            continue
        try:
            value = json.loads(stdout.read_text())
        except ValueError:
            continue
        ref = value.get("attempt_receipt_ref") or {}
        if value.get("target_date") != compact or not ref.get("path"):
            continue
        attempt = Path(ref["path"])
        if not attempt.is_file() or sha256(attempt) != ref.get("sha256"):
            continue
        body = json.loads(attempt.read_text())
        if body.get("core_blockers"):
            continue
        candidates.append((attempt, ref["sha256"], body))
    if not candidates:
        raise Blocked(f"MAINTENANCE_NOT_COMPLETE:{compact}")
    attempt, attempt_sha, body = candidates[-1]
    calendar = body["close_session_receipt_ref"]
    calendar_path = Path(calendar["path"])
    if sha256(calendar_path) != calendar["sha256"]:
        raise Blocked("CALENDAR_RECEIPT_SHA_MISMATCH")
    return attempt, attempt_sha, calendar_path, calendar["sha256"]


def owner_trades(day: str) -> list:
    path = TRADE_INBOX / f"{day.replace('-', '')}.json"
    if not path.exists():
        return []
    value = json.loads(path.read_text())
    trades = value.get("trades")
    if not isinstance(trades, list):
        raise Blocked(f"TRADE_INBOX_INVALID:{path}")
    return trades


def no_event_policy_active() -> None:
    policy = json.loads((WORKSPACE / NO_EVENT_POLICY).read_text())
    if (
        policy.get("revoked_at") is not None
        or policy.get("policy_id") != "cn-daily-no-event-default-v1"
    ):
        raise Blocked("NO_EVENT_DEFAULT_POLICY_NOT_ACTIVE")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trade-date", help="YYYY-MM-DD; default is today in Asia/Shanghai")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    day = args.trade_date or datetime.now(SHANGHAI).date().isoformat()
    date.fromisoformat(day)
    compact = day.replace("-", "")
    steps: list[dict] = []
    receipt = {
        "schema": "cn-evening-close-receipt.v1",
        "trade_date": day,
        "execute": args.execute,
        "started_at": datetime.now(SHANGHAI).isoformat(),
        "steps": steps,
        "broker_calls": False,
        "order_calls": False,
    }
    status = "COMPLETED" if args.execute else "PLANNED"

    def step(name: str, value: dict) -> dict:
        steps.append({"step": name, "result": value})
        return value

    manage = [str(RELEASE_PYTHON), str(RELEASE_CHECKOUT / "scripts/manage_cn_strategy_records.py")]
    try:
        attempt, attempt_sha, calendar, calendar_sha = maintenance_attempt(day)
        step(
            "maintenance",
            {
                "attempt": str(attempt),
                "attempt_sha256": attempt_sha,
                "calendar_receipt": str(calendar),
                "calendar_sha256": calendar_sha,
            },
        )
        trades = owner_trades(day)
        if trades:
            raise Blocked(f"OWNER_TRADES_PRESENT_REQUIRE_TRADE_WRITER:{len(trades)}")
        no_event_policy_active()
        step("owner_trades", {"inbox_trades": 0, "default": "SEVEN_DOMAIN_CLOSED_EMPTY"})

        # 1. Benchmark: append closes after the current generation's end date.
        pointer = WORKSPACE / "data/parquet/cn/benchmarks/_latest.json"
        current_end = json.loads(pointer.read_text())["end_date"]
        if current_end < day:
            start = date.fromordinal(date.fromisoformat(current_end).toordinal() + 1).isoformat()
            argv = [
                str(RELEASE_PYTHON),
                str(RELEASE_CHECKOUT / "scripts/operations/run_cn_benchmark_close.py"),
                "--workspace-root",
                str(WORKSPACE),
                "--start-date",
                start,
                "--end-date",
                day,
                "--generation-id",
                f"benchmark-{compact}-daily-close-v1",
                "--expected-pointer-sha256",
                sha256(pointer),
            ]
            step(
                "benchmark",
                run_json(
                    argv + (["--execute"] if args.execute else []), cwd=WORKSPACE, label="benchmark"
                ),
            )
        else:
            step("benchmark", {"status": "NO_ACTION", "end_date": current_end})

        # 2. Same-day event closure under the standing close policy.
        event_pointer = RECORD_ROOT / "_event_store/current.v1.json"
        closed = json.loads(event_pointer.read_text())["trade_dates"]
        if day in closed:
            step("event_closure", {"status": "NO_ACTION"})
        elif args.execute:
            step(
                "event_closure",
                run_json(
                    manage
                    + [
                        "publish-daily-event-closure",
                        "--record-root",
                        str(RECORD_ROOT),
                        "--project-root",
                        str(WORKSPACE),
                        "--trade-date",
                        day,
                        "--policy-path",
                        CLOSE_POLICY,
                        "--policy-sha256",
                        sha256(WORKSPACE / CLOSE_POLICY),
                        "--maintenance-receipt",
                        str(attempt),
                        "--maintenance-receipt-sha256",
                        attempt_sha,
                        "--expected-event-pointer-sha256",
                        sha256(event_pointer),
                        "--generation-id",
                        f"event-close-{compact}-daily-policy-v1",
                    ],
                    cwd=WORKSPACE,
                    label="event_closure",
                ),
            )
        else:
            step("event_closure", {"status": "WOULD_PUBLISH", "trade_date": day})

        if not args.execute:
            receipt["status"] = status
            return finish(receipt, compact)

        # 3. Official close: plan, prepare, execute exactly the prepared plan, recheck.
        def close_args() -> list[str]:
            return manage + [
                "close-through-latest",
                "--record-root",
                str(RECORD_ROOT),
                "--project-root",
                str(WORKSPACE),
                "--expected-pointer-sha",
                sha256(RECORD_ROOT / "_record_store/current.v1.json"),
                "--expected-market-pointer-sha",
                sha256(WORKSPACE / "data/parquet/cn/_latest.json"),
                "--expected-benchmark-pointer-sha",
                sha256(pointer),
                "--expected-event-pointer-sha",
                sha256(event_pointer),
                "--calendar-receipt",
                str(calendar),
                "--calendar-receipt-sha",
                calendar_sha,
                "--policy-path",
                CLOSE_POLICY,
                "--policy-sha",
                sha256(WORKSPACE / CLOSE_POLICY),
            ]

        plan = step("close_plan", run_json(close_args(), cwd=WORKSPACE, label="close_plan"))
        if plan.get("missing_dates"):
            prepared = step(
                "close_prepare",
                run_json(close_args() + ["--prepare"], cwd=WORKSPACE, label="close_prepare"),
            )
            step(
                "close_execute",
                run_json(
                    close_args() + ["--execute", "--expected-plan-sha", prepared["plan_sha256"]],
                    cwd=WORKSPACE,
                    label="close_execute",
                ),
            )
        recheck = step(
            "close_recheck", run_json(close_args(), cwd=WORKSPACE, label="close_recheck")
        )
        if recheck.get("missing_dates") or recheck.get("last_official_date") != day:
            raise Blocked(f"OFFICIAL_CLOSE_NOT_CAUGHT_UP:{recheck.get('last_official_date')}")

        # 4. Store and accounting verification.
        step(
            "store_verify",
            run_json(
                manage + ["verify", "--record-root", str(RECORD_ROOT)],
                cwd=WORKSPACE,
                label="store_verify",
            ),
        )
        step(
            "accounting_verify",
            run_json(
                [
                    str(RELEASE_PYTHON),
                    str(RELEASE_CHECKOUT / "scripts/prepare_cn_strategy_accounting.py"),
                    "--record-root",
                    str(RECORD_ROOT),
                    "--verify",
                ],
                cwd=WORKSPACE,
                label="accounting_verify",
            ),
        )

        # 5. Dashboard (workspace scripts, private outputs only).
        step(
            "dashboard_export",
            run_json(
                [str(WORKSPACE_PYTHON), "scripts/export_cn_aggressive_dashboard_data.py"],
                cwd=WORKSPACE,
                label="dashboard_export",
            ),
        )
        step(
            "dashboard_check",
            run_json(
                [str(WORKSPACE_PYTHON), "scripts/check_cn_dashboard_export.py"],
                cwd=WORKSPACE,
                label="dashboard_check",
            ),
        )
    except (Blocked, OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        status = "BLOCKED"
        receipt["blocker"] = str(exc)
    receipt["status"] = status
    return finish(receipt, compact)


def finish(receipt: dict, compact: str) -> int:
    receipt["finished_at"] = datetime.now(SHANGHAI).isoformat()
    directory = RECEIPT_ROOT / compact
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(SHANGHAI).strftime("%H%M%S")
    kind = "execute" if receipt["execute"] else "plan"
    path = directory / f"{kind}-{stamp}.json"
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "trade_date": receipt["trade_date"],
                "blocker": receipt.get("blocker"),
                "receipt": str(path),
            },
            ensure_ascii=False,
        )
    )
    return 0 if receipt["status"] in {"COMPLETED", "PLANNED"} else 2


if __name__ == "__main__":
    sys.exit(main())
