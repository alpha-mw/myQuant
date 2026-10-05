#!/usr/bin/env python3
"""Order planning for the automatic Paper account (read-only).

Reads the registered Paper account (its own positions, cash and lots), the sealed
owner risk policies and the Paper sell-signal rules, and prints the orders that
would be submitted for the next CN open session. Positions come from the account,
not the manual strategy ledger: after the first fill the two diverge. Writes only a private receipt under
``data/private/paper_shadow/``; it never touches the strategy record, the Paper
account, the owner-trades inbox or any production pointer.

Shadow mode exists so the owner can check the proposed orders before the Paper
account is seeded and automatic execution is enabled.
"""

from __future__ import annotations

import argparse
import hashlib
import hashlib
import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
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
ACCOUNT_ID = "aggressive-tech-manufacturing-paper-v1"
SHADOW_ROOT = WORKSPACE / "data/private/paper_shadow"
SHANGHAI = ZoneInfo("Asia/Shanghai")


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def paper_account(account_id: str) -> dict:
    from quant_investor.paper.store import PaperStore

    return PaperStore(WORKSPACE).load_account(account_id)


def latest_closed_session() -> str:
    pointer = json.loads((WORKSPACE / "data/parquet/cn/_latest.json").read_text())
    return str(pointer["latest_available_trade_date"]).replace("-", "")


def owner_stops() -> dict:
    from quant_investor.paper.planning import owner_stops as _stops

    return _stops(WORKSPACE)["stops"]


POOL_ROOT = WORKSPACE / "results/intelligence/research_pool/aggressive_tech_manufacturing"
ENTRY_MINIMUM_COMBINED_PERCENTILE = Decimal("0.90")
# The sealed research pool ranks the whole market on two price/volume factors and
# is NOT restricted to the strategy's technology themes: on 2026-09-30 its top
# names were 商贸代理/服饰/家居用品/公路物流/医药商业. Entries therefore require
# per-symbol technology-theme membership evidence, which the pipeline does not
# publish yet, so the entry lane is closed rather than buying off-strategy names.
ENTRY_REQUIRES_THEME_EVIDENCE = True
SEALED_POOL_SOURCE = "SEALED_RESEARCH_POOL_COMBINED_PERCENTILE"
SEALED_TECHNOLOGY_UNIVERSE = "SEALED_TECHNOLOGY_THEME_UNIVERSE"


def sealed_entry_source() -> str:
    """The candidate source the sealed entry policy authorizes."""

    policy = json.loads(
        (
            WORKSPACE
            / "results/policies/paper/aggressive_tech_manufacturing"
            / "owner-paper-risk-execution-policy-20261005-v4.json"
        ).read_text()
    )
    return str((policy.get("entry_policy") or {}).get("candidate_source"))


ENTRY_MAXIMUM_HOLDINGS = 7
ENTRY_MAXIMUM_NEW_PER_WEEK = 2
# Owner quality gates (policy v4 `entry_policy.quality_gates`).
ENTRY_MINIMUM_TURNOVER_WAN = 2000.0
ENTRY_MINIMUM_LISTED_DAYS = 180
ENTRY_TARGET_WEIGHT = "0.14"
ENTRY_MINIMUM_CASH_FRACTION = "0.05"


def _week(session: str) -> tuple[int, int]:
    from datetime import date

    parsed = date(int(session[:4]), int(session[4:6]), int(session[6:]))
    iso = parsed.isocalendar()
    return iso[0], iso[1]


def quality_gates(workspace: Path, trade_date: str, symbols: set[str]) -> dict[str, set[str]]:
    """Owner quality gates: no ST, no recent listing, minimum daily turnover.

    Measured on 2026-09-30: the factor pair rewards low dollar volume, so the
    universe's top names are thin. A 20,000,000 CNY daily floor keeps 39 of the
    52 candidates at or above the 0.90 percentile and drops only the untradeable
    tail; a 50,000,000 floor leaves none.
    """

    import pandas as pd

    pointer = json.loads((workspace / "data/parquet/cn/_latest.json").read_text())
    table = workspace / "data/parquet/cn/_snapshots" / pointer["snapshot_id"] / "table"
    files = sorted((table / "bars").glob("year=*/month=*/*.parquet"))
    if not files:
        raise SystemExit("no strict bars for the quality gates")
    frame = pd.concat([pd.read_parquet(path) for path in files])
    sessions = sorted(day for day in frame.trade_date.unique() if day <= trade_date)
    if not sessions or sessions[-1] != trade_date:
        raise SystemExit(f"no session {trade_date} in the strict bars")
    window = sessions[-20:]
    recent = frame[frame.trade_date.isin(window)]
    turnover = recent.groupby("ts_code").amount.mean() / 10  # 千元 → 万元
    universe = {
        row["symbol"]: row
        for row in json.loads(
            (workspace / "data/cn_universe/stock_basic_membership_latest.json").read_text()
        )["records"]
    }
    from datetime import date

    as_of = date(int(trade_date[:4]), int(trade_date[4:6]), int(trade_date[6:]))
    keep: set[str] = set()
    dropped: dict[str, set[str]] = {"ST": set(), "RECENT_LISTING": set(), "ILLIQUID": set()}
    for symbol in symbols:
        row = universe.get(symbol)
        if row is None:
            dropped["RECENT_LISTING"].add(symbol)
            continue
        if "ST" in (row.get("name") or "").upper():
            dropped["ST"].add(symbol)
            continue
        listed = row.get("list_date") or ""
        if len(listed) == 8:
            listed_date = date(int(listed[:4]), int(listed[4:6]), int(listed[6:]))
            if (as_of - listed_date).days < ENTRY_MINIMUM_LISTED_DAYS:
                dropped["RECENT_LISTING"].add(symbol)
                continue
        if float(turnover.get(symbol, 0.0)) < ENTRY_MINIMUM_TURNOVER_WAN:
            dropped["ILLIQUID"].add(symbol)
            continue
        keep.add(symbol)
    return {"keep": keep, "dropped": dropped}


def latest_pool() -> tuple[dict, dict]:
    """The newest sealed research pool at or before the current session."""

    days = sorted(path for path in POOL_ROOT.glob("*") if path.is_dir())
    if not days:
        raise SystemExit("no sealed research pool is published")
    rank_path = days[-1] / "factor_research_rank.json"
    rank = json.loads(rank_path.read_text())
    return rank["payload"], {
        "path": str(rank_path.relative_to(WORKSPACE)),
        "sha256": hashlib.sha256(rank_path.read_bytes()).hexdigest(),
    }


THEME_REPLAY_ROOT = WORKSPACE / "data/private/intelligence_sources/theme/replays"


def _verified_json(path: Path, expected_sha: str, label: str) -> dict:
    raw = path.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected_sha:
        raise SystemExit(f"{label} SHA differs: {observed}")
    return json.loads(raw)


def theme_membership(*, trade_date: str) -> dict[str, list[str]] | None:
    """Per-symbol technology-theme hits from the sealed theme replay, or None.

    Reads the newest replay receipt for the session, verifies its DC capture ref
    and then the capture's own partition refs, and intersects each captured
    company with the research policy's `technology_theme_ids`. None means no
    replay exists for that session, which keeps the entry lane closed.
    """

    receipts = sorted(THEME_REPLAY_ROOT.glob(f"theme-replay-{trade_date}-*.json"))
    if not receipts:
        return None
    receipt = json.loads(receipts[-1].read_text())
    if receipt.get("trade_date") != trade_date or receipt.get("status") != "COMPLETE":
        return None
    capture = _verified_json(
        WORKSPACE / receipt["dc_capture_ref"]["path"],
        receipt["dc_capture_ref"]["sha256"],
        "theme dc capture",
    )
    policy = json.loads(
        (WORKSPACE / "results/policies/research/aggressive_tech_manufacturing/v2.json").read_text()
    )["payload"]
    technology = set(policy["technology_theme_ids"])
    membership: dict[str, set[str]] = {}
    capture_root = (WORKSPACE / receipt["dc_capture_ref"]["path"]).parent
    for row in capture["partition_rows"]:
        ref = row["partition_capture_ref"]
        part = _verified_json(
            capture_root / "partitions" / f"{row['partition_ordinal']:05d}.json",
            ref["byte_sha256"],
            "theme partition",
        )
        if part.get("api_name") != "dc_member":
            continue
        for member in part.get("rows") or []:
            symbol = member.get("con_code")
            if type(symbol) is str:
                membership.setdefault(symbol, set()).add(member["ts_code"])
    return {symbol: sorted(themes & technology) for symbol, themes in membership.items()}


def entry_orders(
    *,
    account_views,
    account,
    session,
    previous_session,
    pool_rows=None,
    themes=None,
    signal_date=None,
) -> list[dict]:
    """Buys for `session`: first session of a week, policy-capped, pool-sourced."""

    if previous_session is None or _week(session) == _week(previous_session):
        return []
    held = {view["symbol"] for view in account_views}
    room = ENTRY_MAXIMUM_HOLDINGS - len(held)
    if room <= 0:
        return []
    # The weekly cap and the holding cap both bind: never exceed seven names.
    limit = min(ENTRY_MAXIMUM_NEW_PER_WEEK, room)
    source = sealed_entry_source()
    if source == SEALED_TECHNOLOGY_UNIVERSE:
        from quant_investor.paper.planning import technology_candidates

        ranked, _refs = technology_candidates(workspace=WORKSPACE, trade_date=signal_date)
        gates = quality_gates(WORKSPACE, signal_date, {row["symbol"] for row in ranked})
        EntryFunnel.update(
            {
                "ranked": len(ranked),
                "after_gates": len(gates["keep"]),
                "gated_out": {name: len(symbols) for name, symbols in gates["dropped"].items()},
            }
        )
        rows = [row for row in ranked if row["symbol"] in gates["keep"]]
        if pool_rows is not None:
            wanted = {r["symbol"] for r in pool_rows}
            rows = [row for row in rows if row["symbol"] in wanted]
        themes = {row["symbol"]: row["technology_theme_ids"] for row in rows}
    elif source == SEALED_POOL_SOURCE:
        if themes is None:
            themes = theme_membership(trade_date=signal_date)
        if ENTRY_REQUIRES_THEME_EVIDENCE and themes is None:
            raise EntryBlocked(
                "ENTRY_THEME_EVIDENCE_MISSING: 候选无法对策略技术主题核验（池为全市场因子排序，"
                "非科技主题池），买入通道保持关闭"
            )
        payload, _ref = latest_pool()
        if payload.get("as_of") and payload["as_of"] > previous_session:
            raise SystemExit("research pool is newer than the signal session")
        rows = payload["pool_rows"] if pool_rows is None else pool_rows
    else:
        raise EntryBlocked(f"ENTRY_CANDIDATE_SOURCE_UNSUPPORTED: {source}")
    candidates = [
        row
        for row in rows
        if row["symbol"] not in held
        and Decimal(row["combined_percentile"]) >= ENTRY_MINIMUM_COMBINED_PERCENTILE
        and themes.get(row["symbol"])
    ]
    candidates.sort(key=lambda row: (-Decimal(row["combined_percentile"]), row["symbol"]))
    if not candidates:
        EntryNotes.append(
            "ENTRY_NO_THEME_QUALIFIED_CANDIDATE: 池内 "
            f"{len(rows)} 只候选中没有命中策略技术主题的标的"
        )
    orders = []
    for row in candidates[:limit]:
        orders.append(
            {
                "symbol": row["symbol"],
                "name": None,
                "side": "BUY",
                "action": "ENTRY",
                "shares": None,
                "of_shares": 0,
                "policy_row": "entry_policy_pool_candidate",
                "reasons": [f"COMBINED_PERCENTILE:{row['combined_percentile']}"],
                "nav_weight": 0.0,
                "estimated_value_cny": 0.0,
                "target_weight": ENTRY_TARGET_WEIGHT,
                "minimum_cash_fraction": ENTRY_MINIMUM_CASH_FRACTION,
                "requested_ratio": "0.00",
                "reason_codes": ["POOL_COMBINED_PERCENTILE_GE_0_90"],
                "combined_percentile": row["combined_percentile"],
            }
        )
    return orders


def next_session() -> str:
    proofs = sorted(CALENDAR_PROOF.glob("*/calendar-future/proofs/*.json"))
    if not proofs:
        raise SystemExit("no future-calendar proof found")
    return json.loads(proofs[-1].read_text())["next_open_session"]


def _number(value):
    if value is None:
        return None
    text = str(value)
    if text in {"", "nan", "None", "NaN"}:
        return None
    return text


class EntryBlocked(RuntimeError):
    """The entry lane cannot run on verifiable evidence."""


# Non-blocking facts about an empty entry lane, reported alongside the plan.
EntryNotes: list[str] = []
EntryFunnel: dict[str, Any] = {}


POLICY_REASON_CODES = {
    "owner_stop_strict_close_breach": "OWNER_STOP_BREACH",
    "profit_giveback_at_least_35_percent": "PROFIT_GIVEBACK_GE_35",
    "profit_giveback_20_to_35_percent_with_deterioration": (
        "PROFIT_GIVEBACK_20_TO_35_WITH_DETERIORATION"
    ),
}


def _assert_trigger_agrees(view: dict, signal: dict) -> None:
    """Fail closed if the sealed calculator and the rule engine disagree.

    The calculator classifies each lane; the rule engine applies the owner policy
    fractions. They must reach the same place, otherwise one of the two inputs has
    drifted and no order may be produced.
    """

    expected = {
        "REDUCTION_REVIEW": {"REDUCE_50"},
        "REVIEW": {"REVIEW_ONLY", "REDUCE_25", "HOLD"},
        "NOT_CONFIGURED": {"HOLD", "EXIT_100"},
    }
    trailing = view["trailing_trigger"]
    if trailing not in expected:
        raise SystemExit(f"{view['symbol']} unknown trailing trigger {trailing}")
    if signal["action"] not in expected[trailing]:
        # The owner materiality floor (policy v3) deliberately holds back a lane
        # the calculator classified on the raw giveback ratio; only that reason may
        # override a REDUCTION_REVIEW.
        floor_override = (
            trailing == "REDUCTION_REVIEW"
            and signal["action"] == "REVIEW_ONLY"
            and signal["policy_row"] == "trailing_peak_profit_below_materiality_floor"
        )
        if not floor_override:
            raise SystemExit(
                f"{view['symbol']} trigger {trailing} disagrees with action {signal['action']}"
            )
    owner_trigger = view["owner_stop_trigger"]
    if owner_trigger == "BREACH" and signal["action"] != "EXIT_100":
        raise SystemExit(f"{view['symbol']} owner stop breached but no exit was produced")
    if owner_trigger not in {"BREACH", "CLEAR", "NOT_CONFIGURED", "WARNING_NOT_BREACH"}:
        raise SystemExit(f"{view['symbol']} unknown owner trigger {owner_trigger}")


def build_orders(rule_inputs: list[dict], stop_policy: dict) -> list[dict]:
    from quant_investor.paper.execution import calculate_sell_shares
    from quant_investor.paper.rules import HOLD, REVIEW_ONLY, evaluate_position

    actions = {"REDUCE_25": "REDUCE_25", "REDUCE_50": "REDUCE_50", "EXIT_100": "EXIT_100"}
    orders = []
    for view in rule_inputs:
        if not view["close"] or view["blockers"]:
            continue
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
                    "peak_price",
                    "review_price",
                    "reduce_price",
                    "deterioration_evidence",
                )
            }
        )
        _assert_trigger_agrees(view, signal)
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
                "requested_ratio": {"REDUCE_25": "0.25", "REDUCE_50": "0.50", "EXIT_100": "1.00"}[
                    signal["action"]
                ],
                "reason_codes": [POLICY_REASON_CODES[signal["policy_row"]]],
            }
        )
    return orders


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the private shadow receipt")
    args = parser.parse_args()

    as_of = latest_closed_session()
    account = paper_account(ACCOUNT_ID)
    stops = owner_stops()
    from quant_investor.paper.planning import position_views, session_dates

    views = position_views(
        workspace=WORKSPACE,
        account=account,
        as_of=as_of,
        dates=session_dates(WORKSPACE, as_of=as_of),
    )
    orders = build_orders(views, stops)
    sessions = session_dates(WORKSPACE, as_of=as_of)
    blocked: str | None = None
    EntryNotes.clear()
    EntryFunnel.clear()
    try:
        entries = entry_orders(
            account_views=views,
            account=account,
            session=next_session(),
            previous_session=sessions[-1] if sessions else None,
            signal_date=as_of,
        )
    except EntryBlocked as exc:
        entries, blocked = [], str(exc)
    orders = orders + entries
    cash = Decimal(str(account["state"]["cash"]))
    market_value = sum(Decimal(str(view["current_value"])) for view in views)
    snapshot = {
        "cash_cny": f"{cash:.4f}",
        "market_value_cny": f"{market_value:.4f}",
        "total_value_cny": f"{cash + market_value:.4f}",
        "valuation_trade_date": as_of,
    }
    session = next_session()
    created_at = datetime.now(SHANGHAI).isoformat(timespec="seconds")

    receipt = {
        "schema_version": "paper-shadow-orders.v1",
        "created_at": created_at,
        "mode": "SHADOW_READ_ONLY",
        "account_id": ACCOUNT_ID,
        "valuation_trade_date": snapshot["valuation_trade_date"],
        "target_session": session,
        "account": snapshot,
        "orders": orders,
        "entry_lane": {
            "blocked": blocked,
            "notes": list(EntryNotes),
            "funnel": dict(EntryFunnel),
        },
        "holdings": [
            {
                "symbol": view["symbol"],
                "name": view["name"],
                "shares": view["shares"],
                "close": view["close"],
                "hard_stop": view["hard_stop"],
                "hard_stop_source": view["hard_stop_source"],
                "giveback_ratio": view["giveback_ratio"],
                "calc_state": view["calculation_state"],
                "trailing_trigger": view["trailing_trigger"],
                "owner_stop_trigger": view["owner_stop_trigger"],
                "blockers": view["blockers"],
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
        plans = {
            "schema_version": "paper-session-plans.v1",
            "signal_date": as_of,
            "eligible_from_trade_date": session,
            "account_id": ACCOUNT_ID,
            "orders": [
                {
                    "symbol": order["symbol"],
                    "action": order["action"],
                    "shares": order["shares"],
                    "side": order["side"],
                    "requested_ratio": order.get("requested_ratio", "0.00"),
                    "reason_codes": order["reason_codes"],
                    **(
                        {
                            "target_weight": order["target_weight"],
                            "minimum_cash_fraction": order["minimum_cash_fraction"],
                        }
                        if order["side"] == "BUY"
                        else {}
                    ),
                    "signal_date": as_of,
                    "eligible_from_trade_date": session,
                    "source_intent_id": (
                        "paper-"
                        + order["symbol"].lower().replace(".", "-")
                        + "-"
                        + as_of
                        + "-"
                        + order["action"].lower().replace("_", "-")
                    ),
                }
                for order in orders
            ],
        }
        plans_path = target / "plans.json"
        plans_path.write_bytes(canonical(plans) + b"\n")
        print(f"wrote {path}")
        print(f"wrote {plans_path}")

    print(json.dumps(receipt, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
