#!/usr/bin/env python3
"""Shadow order generation for the automatic Paper account (read-only).

Reads the active sealed strategy record, the sealed owner risk policies and the
Paper sell-signal rules, and prints the orders that would be submitted for the
next CN open session. Writes only a private receipt under
``data/private/paper_shadow/``; it never touches the strategy record, the Paper
account, the owner-trades inbox or any production pointer.

Shadow mode exists so the owner can check the proposed orders before the Paper
account is seeded and automatic execution is enabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

WORKSPACE = Path("/Users/maxwell/mySpace/myQuant")
RECORD_POINTER = (
    WORKSPACE
    / "results/strategy_records/CN/aggressive_tech_manufacturing/_record_store/current.v1.json"
)
OWNER_STOP = (
    WORKSPACE
    / "results/policies/risk/aggressive_tech_manufacturing/initial-risk-stop.v1"
    / "owner-stop-policy-20260828-v1.json"
)
CALENDAR_PROOF = WORKSPACE / "results/operations/daily_production/CN"
SHADOW_ROOT = WORKSPACE / "data/private/paper_shadow"
SHANGHAI = ZoneInfo("Asia/Shanghai")


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def active_ledger() -> tuple[Path, dict]:
    pointer = json.loads(RECORD_POINTER.read_text())
    record_id = pointer["active_record_id"]
    record_dir = RECORD_POINTER.parent.parent / record_id
    ledger = record_dir / "ledger_after_manual_switch.parquet"
    if not ledger.exists():
        raise SystemExit(f"ledger missing for active record {record_id}")
    return record_dir, pointer


def owner_stops() -> dict:
    policy = json.loads(OWNER_STOP.read_text())
    return {
        row["symbol"]: {
            "stop": row["initial_stop_price_cny"],
            "source": row["stop_policy_ref"],
        }
        for row in policy["stops"]
        if row.get("initial_stop_state") == "CONFIRMED"
    }


def next_session() -> str:
    proofs = sorted(CALENDAR_PROOF.glob("*/calendar-future/proofs/*.json"))
    if not proofs:
        raise SystemExit("no future-calendar proof found")
    latest = json.loads(proofs[-1].read_text())
    return latest["next_open_session"]


def _number(value):
    if value is None:
        return None
    text = str(value)
    if text in {"", "nan", "None", "NaN"}:
        return None
    return text


def position_views(ledger_path: Path, stops: dict) -> list[dict]:
    import pandas as pd

    frame = pd.read_parquet(ledger_path)
    views = []
    for row in frame.to_dict("records"):
        symbol = row["symbol"]
        owner = stops.get(symbol)
        hard_stop = owner["stop"] if owner else _number(row.get("stage_stop_price"))
        views.append(
            {
                "symbol": symbol,
                "name": row.get("name"),
                "shares": int(row["shares"]),
                "settled_shares": int(row["shares"]),
                "avg_cost": f"{float(row['avg_cost']):.6f}",
                "close": f"{float(row['current_price']):.2f}",
                "hard_stop": hard_stop,
                "hard_stop_source": owner["source"] if owner else "ledger:stage_stop_price",
                "giveback_ratio": _number(row.get("profit_giveback_ratio")),
                "review_price": _number(row.get("trailing_profit_review_price")),
                "reduce_price": _number(row.get("trailing_profit_reduce_price")),
                "deterioration_evidence": [],
                "nav_weight": float(row.get("nav_weight") or 0.0),
                "current_value": float(row.get("current_value") or 0.0),
                "thesis_status": row.get("thesis_status"),
            }
        )
    return views


def account_snapshot(record_dir: Path) -> dict:
    import csv

    summary = record_dir / "pnl_summary.csv"
    with summary.open() as handle:
        row = list(csv.DictReader(handle))[-1]
    return {
        "cash_cny": row["cash_after"],
        "market_value_cny": row["market_value_after"],
        "total_value_cny": row["total_value_after"],
        "valuation_trade_date": row["quote_snapshot"],
    }


def build_orders(rule_inputs: list[dict], stop_policy: dict) -> list[dict]:
    from quant_investor.paper.execution import calculate_sell_shares
    from quant_investor.paper.rules import HOLD, REVIEW_ONLY, evaluate_position

    actions = {"REDUCE_25": "REDUCE_25", "REDUCE_50": "REDUCE_50", "EXIT_100": "EXIT_100"}
    orders = []
    for view in rule_inputs:
        signal = evaluate_position(
            {
                k: view[k]
                for k in (
                    "symbol",
                    "shares",
                    "settled_shares",
                    "avg_cost",
                    "close",
                    "hard_stop",
                    "hard_stop_source",
                    "giveback_ratio",
                    "review_price",
                    "reduce_price",
                    "deterioration_evidence",
                )
            }
        )
        if signal["action"] in (HOLD, REVIEW_ONLY):
            continue
        if signal["action"] not in actions:
            raise SystemExit(f"unexpected action {signal['action']}")
        shares = calculate_sell_shares(
            action=signal["action"], settled_shares=view["settled_shares"]
        )
        if shares == 0:
            continue
        orders.append(
            {
                "symbol": view["symbol"],
                "name": view["name"],
                "side": "SELL",
                "action": signal["action"],
                "shares": shares,
                "of_shares": view["shares"],
                "policy_row": signal["policy_row"],
                "reasons": signal["reasons"],
                "nav_weight": view["nav_weight"],
                "estimated_value_cny": round(shares * float(view["close"]), 2),
            }
        )
    return orders


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the private shadow receipt")
    args = parser.parse_args()

    record_dir, pointer = active_ledger()
    stops = owner_stops()
    views = position_views(record_dir / "ledger_after_manual_switch.parquet", stops)
    orders = build_orders(views, stops)
    snapshot = account_snapshot(record_dir)
    session = next_session()
    created_at = datetime.now(SHANGHAI).isoformat(timespec="seconds")

    receipt = {
        "schema_version": "paper-shadow-orders.v1",
        "created_at": created_at,
        "mode": "SHADOW_READ_ONLY",
        "active_record_id": pointer["active_record_id"],
        "valuation_trade_date": snapshot["valuation_trade_date"],
        "target_session": session,
        "account": snapshot,
        "orders": orders,
        "holdings": [
            {
                "symbol": view["symbol"],
                "name": view["name"],
                "shares": view["shares"],
                "close": view["close"],
                "hard_stop": view["hard_stop"],
                "hard_stop_source": view["hard_stop_source"],
                "giveback_ratio": view["giveback_ratio"],
                "nav_weight": round(view["nav_weight"], 6),
                "thesis_status": view["thesis_status"],
            }
            for view in views
        ],
        "authority": {
            "broker": False,
            "live_order": False,
            "actual_holdings_mutation": False,
            "paper_account_write": False,
            "strategy_record_write": False,
        },
    }
    receipt["receipt_sha256"] = hashlib.sha256(canonical(receipt)).hexdigest()

    if args.write:
        target = SHADOW_ROOT / created_at[:10] / created_at[11:19]
        target.mkdir(parents=True, exist_ok=True)
        path = target / "paper-shadow-orders.v1.json"
        path.write_bytes(canonical(receipt) + b"\n")
        print(f"wrote {path}")

    print(json.dumps(receipt, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
