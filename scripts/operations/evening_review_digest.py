#!/usr/bin/env python3
"""Read-only digest of one evening's CN pipeline for the Hermes evening review.

Collects, without any writes, provider call or launcher start: the 20:25 launcher
slot and its maintenance attempt, the factor-loop state, the target date's DAG
nodes, the evening-close receipts, the Dashboard portfolio/positions/risks, the
Top100 head and today's alerts. Prints one JSON object. Missing inputs are
reported as ``MISSING`` instead of being inferred.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

WORKSPACE = Path("/Users/maxwell/mySpace/myQuant")
MAINTENANCE = WORKSPACE / "data/private/cn_daily_maintenance"
EVENING = WORKSPACE / "data/private/cn_evening_close"
DASHBOARD = WORKSPACE / "portfolio_dashboard/private/generated/cn_aggressive_dashboard.v1.json"
JOURNAL = WORKSPACE / "results/operations/daily_production/CN"
POOL = WORKSPACE / "results/intelligence/research_pool/aggressive_tech_manufacturing"
ALERTS = WORKSPACE / "logs/alerts.jsonl"
SHANGHAI = ZoneInfo("Asia/Shanghai")
MISSING = "MISSING"


def _json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def launcher(compact: str) -> dict:
    slots = sorted((MAINTENANCE / "launcher_attempts").glob(f"slot-2020-{compact}T*"))
    if not slots:
        return {"status": MISSING}
    slot = slots[-1]
    ended = _json(slot / "ended.json") or {}
    result = {"slot": slot.name, "exit_code": ended.get("process_exit_code", MISSING)}
    stdout = _json(slot / "maintenance.stdout.json") or {}
    ref = (stdout.get("attempt_receipt_ref") or {}).get("path")
    attempt = _json(Path(ref)) if ref else None
    if attempt is None:
        result["attempt"] = MISSING
        return result
    result["attempt"] = {
        "path": ref,
        "target_date": attempt.get("target_date"),
        "status": attempt.get("status"),
        "core_blockers": attempt.get("core_blockers"),
        "blockers": attempt.get("blockers"),
        "stages": {s["stage"]: s["status"] for s in attempt.get("stage_results", [])},
    }
    return result


def factor_state() -> dict:
    state = _json(MAINTENANCE / "factor-loop-state.json")
    if state is None:
        return {"status": MISSING}
    return {k: state.get(k) for k in ("trade_date", "phase", "context_sha256")}


def dag(target: str | None) -> dict:
    status = _json(JOURNAL / target / "dag-status.v1.json") if target else None
    if status is None:
        return {"status": MISSING}
    return {
        "status": status.get("status"),
        "nodes": {
            name: node.get("command_status") or node.get("blocking_reason")
            for name, node in status.get("nodes", {}).items()
        },
    }


def evening_close(compact: str) -> list:
    rows = []
    for path in sorted((EVENING / compact).glob("execute-*.json")):
        receipt = _json(path) or {}
        rows.append(
            {
                "receipt": path.name,
                "status": receipt.get("status"),
                "blocker": receipt.get("blocker"),
                "steps": {s["step"]: s["result"].get("status") for s in receipt.get("steps", [])},
            }
        )
    return rows


def dashboard() -> dict:
    board = _json(DASHBOARD)
    if board is None:
        return {"status": MISSING}
    portfolio = board.get("portfolio", {})
    return {
        "status": board.get("status"),
        "latest_data_date": board.get("latest_data_date"),
        "portfolio": {
            k: portfolio.get(k)
            for k in (
                "adjusted_total_value",
                "cash_weight",
                "cumulative_return",
                "latest_record_interval_return",
                "max_drawdown",
                "current_unrealized_pnl",
            )
        },
        "positions": [
            {
                k: p.get(k)
                for k in (
                    "symbol",
                    "name",
                    "shares",
                    "avg_cost",
                    "recorded_price",
                    "nav_weight",
                    "unrealized_pnl",
                    "thesis_status",
                )
            }
            for p in board.get("positions", [])
        ],
        "risks": board.get("risks"),
        "warnings": board.get("warnings"),
    }


def top100(target: str | None, head: int) -> dict:
    if not target:
        return {"status": MISSING}
    day = f"{target[:4]}-{target[4:6]}-{target[6:]}"
    path = POOL / day / "top100.parquet"
    if not path.is_file():
        return {"status": MISSING, "path": str(path)}
    import pandas as pd

    frame = pd.read_parquet(path).sort_values("rank").head(head)
    return {"path": str(path), "head": frame.astype(str).to_dict(orient="records")}


def alerts(day: str) -> list:
    if not ALERTS.is_file():
        return []
    rows = []
    for line in ALERTS.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if str(row.get("at", "")).startswith(day):
            rows.append(row)
    return rows


def digest(day: str, head: int = 20) -> dict:
    compact = day.replace("-", "")
    slot = launcher(compact)
    attempt = slot.get("attempt")
    state = factor_state()
    target = attempt.get("target_date") if isinstance(attempt, dict) else None
    target = target or state.get("trade_date")
    return {
        "schema": "cn-evening-review-digest.v1",
        "date": day,
        "generated_at": datetime.now(SHANGHAI).isoformat(timespec="seconds"),
        "read_only": True,
        "launcher": slot,
        "factor_state": state,
        "target_trade_date": target or MISSING,
        "dag": dag(target),
        "evening_close": evening_close(compact),
        "dashboard": dashboard(),
        "top100": top100(target, head),
        "alerts": alerts(day),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="YYYY-MM-DD; default is today in Asia/Shanghai")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    day = args.date or datetime.now(SHANGHAI).date().isoformat()
    print(json.dumps(digest(day, args.top), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
