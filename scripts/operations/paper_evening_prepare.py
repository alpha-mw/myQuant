#!/usr/bin/env python3
"""Evening run for the Paper account: evidence, orders, fills, next plans.

Runs after the CN evening close for session `D`:

1. capture `D`'s exchange price limits (one provider request, sealed under
   `data/private/paper_evidence/`);
2. for each plan written for `D`, emit the intent + eligibility files and **fill
   it** through the writer (the owner delegated paper execution, so there is no
   confirmation step);
3. write the plans for the next session from `D`'s sealed record.

Orders are produced and filled one at a time because every fill advances the
account pointer, and each intent binds the pointer it was built against. A fill
that pends (a limit-up open, a suspension, a pending corporate action) is
reported with its named blocker and the run continues; an expired intent is
reported as such, never retried silently.
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
PAPER_RELEASE = WORKSPACE / "operations/releases/paper.env"
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


def paper_release() -> dict[str, str]:
    """The frozen release the Paper writer runs from (writer verifies it)."""

    values: dict[str, str] = {}
    for line in PAPER_RELEASE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    required = {
        "RELEASE_INSTALL_DIR",
        "RELEASE_CHECKOUT_DIR",
        "RELEASE_INSTALL_INPUT",
        "RELEASE_INSTALL_INPUT_SHA256",
    }
    if not required.issubset(values):
        raise SystemExit("operations/releases/paper.env is incomplete")
    return values


def _writer(release: dict[str, str], command: str, *args: str) -> tuple[int, str]:
    environment = {"PYTHONPATH": "", "HOME": str(Path.home())}
    import os

    completed = subprocess.run(
        [
            str(Path(release["RELEASE_INSTALL_DIR"]) / "bin/python"),
            "-I",
            "-m",
            "quant_investor",
            "paper",
            command,
            "--workspace-root",
            str(WORKSPACE),
            "--account-id",
            ACCOUNT_ID,
            *args,
            "--release-install-input",
            release["RELEASE_INSTALL_INPUT"],
            "--expected-release-install-input-sha256",
            release["RELEASE_INSTALL_INPUT_SHA256"],
            "--release-repository-root",
            release["RELEASE_CHECKOUT_DIR"],
        ],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
        env={**os.environ, **environment},
    )
    return completed.returncode, (completed.stdout + completed.stderr).strip()


def _fill(release: dict[str, str], item: dict, order: dict, pointer: str) -> dict:
    """Fill one prepared order; the writer is the only mutation surface."""

    command = (
        "owner-run"
        if order.get("price_basis")
        else "entry-run" if order.get("side") == "BUY" else "risk-exit-run"
    )
    code, output = _writer(
        release,
        command,
        "--intent",
        item["intent_ref"]["path"],
        "--expected-intent-sha256",
        item["intent_ref"]["sha256"],
        "--eligibility",
        item["eligibility_ref"]["path"],
        "--expected-eligibility-sha256",
        item["eligibility_ref"]["sha256"],
        "--expected-current-pointer-sha256",
        pointer,
        "--allow-write",
    )
    if code != 0:
        return {"symbol": item["symbol"], "status": "BLOCKED", "output": output[-400:]}
    try:
        result = json.loads(output)
    except ValueError:
        return {"symbol": item["symbol"], "status": "UNPARSABLE", "output": output[-400:]}
    return {
        "symbol": item["symbol"],
        "command": command,
        "status": result.get("command_status"),
        "sequence": result.get("sequence"),
        "record_path": result.get("record_path"),
    }


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


def applied_intents() -> set[str]:
    """Source intents the account has already applied (filled or terminal)."""

    from quant_investor.paper.store import PaperStore

    store = PaperStore(WORKSPACE)
    if ACCOUNT_ID not in store.account_ids():
        return set()
    loaded = store.load_account(ACCOUNT_ID)
    return set((loaded["state"].get("applied_source_intents") or {}).keys())


def plans_for(session: str, *, applied: set[str] | None = None) -> list[dict]:
    """Plans that are due and not yet applied, oldest first.

    An order planned for an earlier session stays in scope until the account has
    applied it: the writer pends what the evidence does not support, and the
    agent must not leave a planned order silently unfilled because one evening's
    data was late.
    """

    applied = applied_intents() if applied is None else applied
    by_parent: dict[str, dict] = {}
    for path in SHADOW_ROOT.glob("*/*/plans.json"):
        try:
            value = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        due = value.get("eligible_from_trade_date")
        if not due or due > session or not value.get("orders"):
            continue
        key = path.parent.as_posix()
        if key not in by_parent or due > by_parent[key]["due"]:
            by_parent[key] = {"due": due, "path": path, "value": value}
    pending: dict[str, dict] = {}
    for entry in sorted(
        by_parent.values(), key=lambda item: (item["due"], item["path"].as_posix())
    ):
        for order in entry["value"]["orders"]:
            if order["source_intent_id"] in applied:
                continue
            # One order identity, whichever plans file announced it first.
            pending.setdefault(
                order["source_intent_id"],
                {"plans": entry["path"], "order": order, "due": entry["due"]},
            )
    return [pending[key] for key in sorted(pending, key=lambda key: (pending[key]["due"], key))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trade-date", required=True, help="YYYYMMDD session")
    parser.add_argument("--plan-only", action="store_true", help="skip capture and order handling")
    parser.add_argument(
        "--no-fill",
        action="store_true",
        help="prepare the intent and eligibility files but do not fill",
    )
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

        code, output = _run(
            "paper_capture_corporate_actions.py", "--trade-date", session, "--write"
        )
        if code != 0:
            report["blockers"].append("CORPORATE_ACTION_CAPTURE_FAILED")
            report["steps"]["capture_corporate_actions"] = {
                "status": "BLOCKED",
                "output": output[-800:],
            }
        else:
            try:
                captured = json.loads(output[output.index("{") :])
            except ValueError:
                captured = {"status": "UNPARSABLE", "output": output[-400:]}
            report["steps"]["capture_corporate_actions"] = {
                "status": captured.pop("status", "CAPTURED"),
                **captured,
            }

        due = plans_for(session)
        if not due:
            report["steps"]["orders"] = {"status": "NO_PLANS"}
        elif report["steps"].get("capture_limits", {}).get("status") != "CAPTURED":
            report["steps"]["orders"] = {"status": "NO_EVIDENCE"}
        else:
            evidence = report["steps"]["capture_limits"]
            release = paper_release()
            filled: list[dict] = []
            failed: list[dict] = []
            work_root = WORKSPACE / "data/private/paper_intents" / session / "orders"
            for index, item in enumerate(due, start=1):
                order = item["order"]
                single = {
                    "schema_version": "paper-session-plans.v1",
                    "signal_date": order["signal_date"],
                    "eligible_from_trade_date": order["eligible_from_trade_date"],
                    "account_id": ACCOUNT_ID,
                    "orders": [order],
                }
                single_path = work_root / f"{index:02d}-{order['symbol']}.json"
                single_path.parent.mkdir(parents=True, exist_ok=True)
                single_path.write_bytes(
                    json.dumps(
                        single, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    ).encode()
                )
                single_path.chmod(0o600)
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
                    single_path.relative_to(WORKSPACE).as_posix(),
                    "--write",
                )
                if code != 0:
                    failed.append(
                        {
                            "symbol": order["symbol"],
                            "status": "PREPARE_FAILED",
                            "output": output[-400:],
                        }
                    )
                    continue
                prepared = json.loads(output)
                for skipped in prepared["skipped"]:
                    failed.append(
                        {
                            "symbol": skipped["symbol"],
                            "status": "SKIPPED",
                            "reason": skipped["reason"],
                        }
                    )
                for entry in prepared["prepared"]:
                    if args.no_fill:
                        filled.append(
                            {
                                "symbol": entry["symbol"],
                                "status": "PREPARED",
                                "intent_ref": entry["intent_ref"]["path"],
                                "eligibility_ref": entry["eligibility_ref"]["path"],
                            }
                        )
                        continue
                    result = _fill(release, entry, order, prepared["pointer_sha256"])
                    if result["status"] in {"FILLED", "NO_ACTION_ALREADY_APPLIED"}:
                        filled.append(result)
                    else:
                        failed.append(result)
            report["steps"]["orders"] = {
                "status": "DONE",
                "due": len(due),
                "filled": filled,
                "failed": failed,
            }
            if failed:
                report["blockers"].append("SOME_ORDERS_NOT_FILLED")

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
                "entry_lane": receipt.get("entry_lane"),
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
