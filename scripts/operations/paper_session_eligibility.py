#!/usr/bin/env python3
"""Build Paper intent + eligibility files for one CN session (fail closed).

This is the evidence layer between the sealed sell/buy rules and the Paper
writer. It reads the current account, the strict session bars and an explicit
price-limit evidence file, then emits one `paper-risk-intent.v1` and one
`paper-input-eligibility.v1` per order, canonical and owner-only, ready for
`paper risk-exit-run`.

Price limits are **never derived here**. A hand-rolled board/ST rule disagrees
with the exchange on ~0.33% of real symbol-days (ST renames, new listings), so
the caller must supply the session's limits (Tushare `stk_limit`) as a sealed
evidence file:

```json
{
  "schema_version": "paper-price-limit-evidence.v1",
  "trade_date": "20261008",
  "source": "tushare.stk_limit",
  "symbols": {
    "601899.SH": {"limit_up": "32.77", "limit_down": "26.81", "previous_close": "29.79"}
  }
}
```

Missing limits, a missing bar, or a bar that contradicts the supplied limits
keeps the symbol out of the emitted set — the writer must never see invented
evidence.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

WORKSPACE = Path("/Users/maxwell/mySpace/myQuant")
RECORD_POINTER = (
    WORKSPACE
    / "results/strategy_records/CN/aggressive_tech_manufacturing/_record_store/current.v1.json"
)
PAPER_ROOT = WORKSPACE / "results/paper/accounts"
SNAPSHOT_POINTER = WORKSPACE / "data/parquet/cn/_latest.json"
OUTPUT_ROOT = WORKSPACE / "data/private/paper_intents"
EVIDENCE_ROOT = WORKSPACE / "data/private/paper_evidence"
SHANGHAI = ZoneInfo("Asia/Shanghai")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _relative(path: Path) -> str:
    return path.relative_to(WORKSPACE).as_posix()


def _write_canonical(path: Path, value: dict) -> dict[str, str]:
    from quant_investor.contracts import canonical_json_bytes

    raw = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(0o600)
    return {"path": _relative(path), "sha256": _sha(raw)}


def session_evidence(trade_date: str) -> dict[str, dict[str, str]]:
    """Resolve the five evidence refs the eligibility contract requires.

    - calendar: the session's sealed calendar compilation from the daily journal
    - bar / suspension: the strict snapshot manifest; a bar for the symbol on the
      session is itself the evidence that it traded, and the manifest binds the
      snapshot those bars came from
    - corporate action: the sealed research risk calculator's verdict, sealed
      here per session
    """

    calendar_refs = sorted(
        (WORKSPACE / "results/operations/daily_production/CN").glob(
            f"{trade_date}/nodes/calendar/*/attempt-*/terminal.json"
        )
    )
    if not calendar_refs:
        raise SystemExit(f"no calendar node terminal for {trade_date}")
    terminal = json.loads(calendar_refs[-1].read_text())
    calendar_ref = dict(terminal["output_refs"]["calendar_compilation_ref"])
    pointer = json.loads(SNAPSHOT_POINTER.read_text())
    manifest = WORKSPACE / pointer["manifest_path"]
    bar_ref = {"path": _relative(manifest), "sha256": _sha(manifest.read_bytes())}
    return {"calendar_ref": calendar_ref, "bar_ref": bar_ref}


def seal_corporate_action_evidence(trade_date: str, monitor: dict) -> dict[str, str]:
    from quant_investor.contracts import canonical_json_bytes

    path = EVIDENCE_ROOT / trade_date / "risk-monitor.json"
    raw = canonical_json_bytes(monitor)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_bytes(raw)
    path.chmod(0o600)
    return {"path": _relative(path), "sha256": _sha(raw)}


def seal_nav_evidence(trade_date: str, state: dict, bars: dict[str, dict]) -> dict[str, str]:
    """Seal the account valuation the entry sizing binds to."""

    from quant_investor.contracts import canonical_json_bytes

    cash = Decimal(str(state["state"]["cash"]))
    positions = []
    market_value = Decimal("0")
    for row in state["ledger"]:
        bar = bars.get(row["symbol"])
        value = None if bar is None else Decimal(bar["open"]) * Decimal(int(row["shares"]))
        if value is not None:
            market_value += value
        positions.append(
            {
                "symbol": row["symbol"],
                "shares": int(row["shares"]),
                "session_open": None if bar is None else bar["open"],
                "session_value_cny": None if value is None else f"{value:.4f}",
            }
        )
    payload = canonical_json_bytes(
        {
            "schema_version": "paper-account-valuation.v1",
            "trade_date": trade_date,
            "cash_cny": f"{cash:.4f}",
            "market_value_cny": f"{market_value:.4f}",
            "nav_cny": f"{cash + market_value:.4f}",
            "positions": positions,
            "basis": "SESSION_OPEN",
        }
    )
    path = EVIDENCE_ROOT / trade_date / "account-valuation.v1.json"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_bytes(payload)
    path.chmod(0o600)
    return {"path": _relative(path), "sha256": _sha(payload)}


def account_state(account_id: str) -> dict:
    from quant_investor.paper.store import PaperStore

    return PaperStore(WORKSPACE).load_account(account_id)


def _snapshot_bars(months: set[str]):
    import pandas as pd

    pointer = json.loads(SNAPSHOT_POINTER.read_text())
    table = WORKSPACE / "data/parquet/cn/_snapshots" / pointer["snapshot_id"] / "table"
    frames = []
    for month in sorted(months):
        year, month_number = month[:4], month[4:6]
        files = sorted(
            (table / "bars" / f"year={year}" / f"month={month_number}").glob("*.parquet")
        )
        frames.extend(pd.read_parquet(path) for path in files)
    return pd.concat(frames) if frames else None


def session_bars(trade_date: str) -> tuple[dict[str, dict], dict[str, str]]:
    """Strict bars for one session plus the adjustment verdict per symbol.

    The adjustment check mirrors the sealed risk calculator: a session whose
    `adj_factor` differs from the previous session's is a corporate action that
    must be reviewed before any fill, so `paper-input-eligibility.v1` never sees
    an assumed CLEAR.
    """

    import pandas as pd

    frame = _snapshot_bars({trade_date[:6]})
    if frame is None:
        return {}, {}
    sessions = sorted(day for day in frame.trade_date.unique() if day <= trade_date)
    if not sessions or sessions[-1] != trade_date:
        return {}, {}
    previous = sessions[-2] if len(sessions) > 1 else None
    rows = frame[frame.trade_date == trade_date]
    bars = {
        row.ts_code: {
            "open": f"{float(row.open):.4f}",
            "previous_close": f"{float(row.pre_close):.4f}",
            "high": f"{float(row.high):.4f}",
            "low": f"{float(row.low):.4f}",
        }
        for row in rows.itertuples(index=False)
    }
    states: dict[str, str] = {}
    if previous is not None:
        prior = {
            row.ts_code: row.adj_factor
            for row in frame[frame.trade_date == previous].itertuples(index=False)
        }
        factors = {row.ts_code: row.adj_factor for row in rows.itertuples(index=False)}
        for symbol, factor in factors.items():
            states[symbol] = "CLEAR" if symbol in prior and prior[symbol] == factor else "PENDING"
    return bars, states


def read_limits(path: Path, expected_sha: str, trade_date: str) -> dict[str, dict]:
    raw = path.read_bytes()
    observed = _sha(raw)
    if observed != expected_sha:
        raise SystemExit(f"limits evidence SHA differs: {observed}")
    value = json.loads(raw)
    if (
        value.get("schema_version") != "paper-price-limit-evidence.v1"
        or value.get("trade_date") != trade_date
        or type(value.get("symbols")) is not dict
    ):
        raise SystemExit("limits evidence schema or trade date differs")
    return value["symbols"]


def corporate_action_states(trade_date: str) -> dict[str, str]:
    """Reuse the sealed research risk calculator's own corporate-action verdicts."""

    import subprocess

    out_dir = Path("/private/tmp/myquant-cn") / f"paper-eligibility-{trade_date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "risk-monitor.json"
    as_of = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
    subprocess.run(
        [
            str(WORKSPACE / ".venv/bin/python"),
            "scripts/export_cn_research_risk.py",
            "--as-of",
            as_of,
            "--output",
            str(output),
        ],
        cwd=WORKSPACE,
        check=True,
        capture_output=True,
    )
    monitor = json.loads(output.read_text())
    states = {}
    for row in monitor["rows"]:
        blocked = bool(row.get("blockers"))
        states[row["symbol"]] = "PENDING" if blocked else "CLEAR"
    return states, monitor


def build(
    *,
    account_id: str,
    trade_date: str,
    orders: list[dict],
    limits: dict[str, dict],
    bars: dict[str, dict],
    corporate: dict[str, str],
    pointer_sha: str,
    policy_ref: dict[str, str],
    policy_id: str,
    account_state: dict,
    nav_ref: dict[str, str],
) -> tuple[list[dict], list[dict]]:
    from quant_investor.paper.contracts import POLICY_ID as _POLICY_ID
    from quant_investor.paper.execution import economic_action_key

    emitted: list[dict] = []
    skipped: list[dict] = []
    positions = {row["symbol"]: row for row in account_state["ledger"]}
    nav = Decimal(str(account_state["state"]["cash"]))
    for row in account_state["ledger"]:
        bar = bars.get(row["symbol"])
        if bar is not None:
            nav += Decimal(bar["open"]) * Decimal(int(row["shares"]))
    for order in sorted(orders, key=lambda item: (item.get("side", "SELL"), item["symbol"])):
        symbol = order["symbol"]
        position = positions.get(symbol)
        side = order.get("side", "SELL")
        if side == "SELL" and position is None:
            skipped.append({"symbol": symbol, "reason": "NOT_HELD"})
            continue
        bar = bars.get(symbol)
        limit = limits.get(symbol)
        if bar is None:
            # No bar for the session: suspended or not yet published.
            skipped.append({"symbol": symbol, "reason": "NO_SESSION_BAR"})
            continue
        if not limit:
            skipped.append({"symbol": symbol, "reason": "NO_LIMIT_EVIDENCE"})
            continue
        if float(limit["limit_down"]) > float(bar["low"]) or float(limit["limit_up"]) < float(
            bar["high"]
        ):
            skipped.append({"symbol": symbol, "reason": "LIMIT_CONTRADICTS_BAR"})
            continue
        if limit.get("previous_close") not in (None, bar["previous_close"]):
            skipped.append({"symbol": symbol, "reason": "PREVIOUS_CLOSE_DRIFT"})
            continue
        economic = economic_action_key(
            account_id=account_id,
            policy_id=policy_id or _POLICY_ID,
            signal_date=order["signal_date"],
            symbol=symbol,
            action=order["action"],
            shares=int(order["shares"]) if order.get("shares") else 0,
        )
        if side == "BUY":
            emitted.append(
                {
                    "symbol": symbol,
                    "side": "BUY",
                    "intent": {
                        "schema_version": "paper-entry-intent.v1",
                        "source_intent_id": order["source_intent_id"],
                        "idempotency_key_sha256": economic,
                        "economic_action_key_sha256": economic,
                        "account_id": account_id,
                        "strategy_id": "aggressive_tech_manufacturing",
                        "signal_date": order["signal_date"],
                        "eligible_from_trade_date": order["eligible_from_trade_date"],
                        "symbol": symbol,
                        "name": order.get("name") or symbol,
                        "side": "BUY",
                        "action": "ENTRY",
                        "target_weight": order["target_weight"],
                        "minimum_cash_fraction": order["minimum_cash_fraction"],
                        "account_nav_cny": f"{nav:.4f}",
                        "nav_evidence_ref": dict(nav_ref),
                        "reason_codes": sorted(order["reason_codes"]),
                        "policy_ref": dict(policy_ref),
                        "expected_account_pointer_sha256": pointer_sha,
                        "expected_position": (
                            {
                                "shares": int(position["shares"]),
                                "settled_shares": int(position["settled_shares"]),
                                "avg_cost": f"{float(position['avg_cost']):.4f}",
                            }
                            if position is not None
                            else None
                        ),
                        "evidence_refs": [],
                        "broker": False,
                        "real_order": False,
                        "actual_holdings_mutation": False,
                    },
                    "bar": bar,
                    "limit": limit,
                    "corporate": corporate.get(symbol, "PENDING"),
                }
            )
            continue
        intent = {
            "schema_version": "paper-risk-intent.v1",
            "source_intent_id": order["source_intent_id"],
            "idempotency_key_sha256": economic,
            "economic_action_key_sha256": economic,
            "account_id": account_id,
            "strategy_id": "aggressive_tech_manufacturing",
            "signal_date": order["signal_date"],
            "eligible_from_trade_date": order["eligible_from_trade_date"],
            "symbol": symbol,
            "action": order["action"],
            "requested_ratio": order["requested_ratio"],
            "requested_shares": int(order["shares"]),
            "reason_codes": sorted(order["reason_codes"]),
            "policy_ref": dict(policy_ref),
            "expected_account_pointer_sha256": pointer_sha,
            "expected_position": {
                "shares": int(position["shares"]),
                "settled_shares": int(position["settled_shares"]),
                "avg_cost": f"{float(position['avg_cost']):.4f}",
            },
            "evidence_refs": [],
            "broker": False,
            "real_order": False,
            "actual_holdings_mutation": False,
        }
        emitted.append(
            {
                "symbol": symbol,
                "intent": intent,
                "bar": bar,
                "limit": limit,
                "corporate": corporate.get(symbol, "PENDING"),
            }
        )
    return emitted, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--limits", required=True)
    parser.add_argument("--expected-limits-sha256", required=True)
    parser.add_argument("--plans", required=True, help="plans.json from the session planner")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    from quant_investor.paper.contracts import (
        POLICY_ID,
        POLICY_RELATIVE_PATH,
        POLICY_SHA256,
        seal_document,
        validate_eligibility,
        validate_entry_intent,
        validate_intent,
    )

    state = account_state(args.account_id)
    limits = read_limits(WORKSPACE / args.limits, args.expected_limits_sha256, args.trade_date)
    bars, adjustment_states = session_bars(args.trade_date)
    corporate, monitor = corporate_action_states(args.trade_date)
    for symbol, adjustment in adjustment_states.items():
        if symbol not in corporate:
            corporate[symbol] = adjustment
        elif corporate[symbol] == "CLEAR" and adjustment == "PENDING":
            corporate[symbol] = "PENDING"
    evidence = session_evidence(args.trade_date)
    corporate_ref = seal_corporate_action_evidence(args.trade_date, monitor)
    nav_ref = seal_nav_evidence(args.trade_date, state, bars)
    plans = json.loads((WORKSPACE / args.plans).read_text())
    policy_ref = {"path": POLICY_RELATIVE_PATH, "sha256": POLICY_SHA256}
    emitted, skipped = build(
        account_id=args.account_id,
        trade_date=args.trade_date,
        orders=plans["orders"],
        limits=limits,
        bars=bars,
        corporate=corporate,
        pointer_sha=state["pointer_sha256"],
        policy_ref=policy_ref,
        policy_id=POLICY_ID,
        account_state=state,
        nav_ref=nav_ref,
    )
    stamp = datetime.now(SHANGHAI).strftime("%Y%m%dT%H%M%S")
    day_root = OUTPUT_ROOT / args.trade_date / stamp
    results = []
    for item in emitted:
        symbol = item["symbol"]
        bar, limit = item["bar"], item["limit"]
        from quant_investor.contracts import canonical_json_bytes

        intent = seal_document(item["intent"])
        if item.get("side") == "BUY":
            validate_entry_intent(intent)
        else:
            validate_intent(intent)
        intent_path = day_root / symbol / "intent.v1.json"
        intent_ref = {
            "path": _relative(intent_path),
            "sha256": _sha(canonical_json_bytes(intent)),
        }
        if args.write:
            intent_ref = _write_canonical(intent_path, intent)
        eligibility = seal_document(
            {
                "schema_version": "paper-input-eligibility.v1",
                "account_id": args.account_id,
                "source_intent_ref": dict(intent_ref),
                "symbol": symbol,
                "signal_date": intent["signal_date"],
                "eligible_trade_date": intent["eligible_from_trade_date"],
                "evaluated_trade_date": args.trade_date,
                "open_price": bar["open"],
                "previous_close": bar["previous_close"],
                "limit_up": f"{float(limit['limit_up']):.4f}",
                "limit_down": f"{float(limit['limit_down']):.4f}",
                "suspended": False,
                "corporate_action_state": item["corporate"],
                "open_session_ordinal": 1,
                "expiry_session_ordinal": 3,
                "calendar_ref": evidence["calendar_ref"],
                "raw_bar_ref": evidence["bar_ref"],
                "price_limit_ref": {
                    "path": args.limits,
                    "sha256": args.expected_limits_sha256,
                },
                # The session bar is also the evidence that the symbol traded.
                "suspension_ref": evidence["bar_ref"],
                "corporate_action_ref": corporate_ref,
                "evidence_status": "READY",
            }
        )
        if eligibility["corporate_action_state"] != "CLEAR":
            skipped.append({"symbol": symbol, "reason": "CORPORATE_ACTION_PENDING"})
            continue
        eligibility_path_target = day_root / symbol / "eligibility.v1.json"
        eligibility_ref = {
            "path": _relative(eligibility_path_target),
            "sha256": _sha(canonical_json_bytes(eligibility)),
        }
        validate_eligibility(eligibility)
        if args.write:
            eligibility_ref = _write_canonical(eligibility_path_target, eligibility)
        results.append(
            {
                "symbol": symbol,
                "side": item.get("side", "SELL"),
                "action": intent["action"],
                "shares": intent.get("requested_shares"),
                "intent_ref": intent_ref,
                "eligibility_ref": eligibility_ref,
            }
        )
    print(
        json.dumps(
            {
                "account_id": args.account_id,
                "trade_date": args.trade_date,
                "pointer_sha256": state["pointer_sha256"],
                "prepared": results,
                "skipped": skipped,
                "write": bool(args.write),
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
