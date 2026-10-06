#!/usr/bin/env python3
"""One sealed decision digest per session for the agent panel.

Every agent in the panel reads this file and nothing else: it is the single,
hashed input, so a memo can be replayed and audited, and no agent can quietly
consult different data. Deterministic and read-only — it computes, it never
decides, and it never writes to the account.

Sections:

- `account`      the registered Paper account's cash, NAV, positions and lots
- `candidates`   the technology universe ranked by the sealed factor formula,
                 with the quality-gate funnel and each name's theme membership
- `risk`         per-position owner stop, moving take-profit ladder and trigger
- `orders`       the orders already planned for the next session
- `evidence`     which evidence lanes exist for the session, with refs, and
                 which are missing (the panel must say INSUFFICIENT_EVIDENCE
                 rather than fill these in)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

WORKSPACE = Path("/Users/maxwell/mySpace/myQuant")
GENERATED = WORKSPACE / "data/private/decision_digests"
ACCOUNT_ID = "aggressive-tech-manufacturing-paper-v1"
VETO_ROOT = WORKSPACE / "data/private/cn_daily_maintenance"
RECORD_POINTER = (
    WORKSPACE
    / "results/strategy_records/CN/aggressive_tech_manufacturing/_record_store/current.v1.json"
)


def _ref(path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(WORKSPACE).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _session() -> str:
    pointer = json.loads((WORKSPACE / "data/parquet/cn/_latest.json").read_text())
    return str(pointer["latest_available_trade_date"]).replace("-", "")


def _account() -> dict:
    from quant_investor.paper.store import PaperStore

    store = PaperStore(WORKSPACE)
    if ACCOUNT_ID not in store.account_ids():
        return {"status": "NOT_REGISTERED"}
    loaded = store.load_account(ACCOUNT_ID)
    return {
        "status": "READY",
        "account_id": ACCOUNT_ID,
        "sequence": loaded["pointer"]["sequence"],
        "pointer_sha256": loaded["pointer_sha256"],
        "cash": str(loaded["state"]["cash"]),
        "realized_pnl": str(loaded["state"]["realized_pnl"]),
        "cumulative_fees": str(loaded["state"]["cumulative_fees"]),
        "applied_source_intents": sorted(loaded["state"].get("applied_source_intents") or {}),
        "positions": [
            {
                "symbol": row["symbol"],
                "name": row["name"],
                "shares": int(row["shares"]),
                "settled_shares": int(row["settled_shares"]),
                "avg_cost": str(row["avg_cost"]),
                "lots": json.loads(row["acquisition_lots_json"]),
            }
            for row in loaded["ledger"]
            if int(row["shares"]) > 0
        ],
    }


def _candidates(session: str) -> dict:
    import importlib.util

    from quant_investor.paper.planning import technology_candidates

    spec = importlib.util.spec_from_file_location(
        "paper_shadow_orders", WORKSPACE / "scripts/operations/paper_shadow_orders.py"
    )
    planner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(planner)

    try:
        ranked, refs = technology_candidates(workspace=WORKSPACE, trade_date=session)
        gates = planner.quality_gates(WORKSPACE, session, {row["symbol"] for row in ranked})
        held = {row["symbol"] for row in _account().get("positions", [])}
        shortlist = [
            {
                "symbol": row["symbol"],
                "combined_percentile": row["combined_percentile"],
                "technology_theme_ids": row["technology_theme_ids"],
            }
            for row in ranked
            if row["symbol"] in gates["keep"]
            and row["symbol"] not in held
            and row["combined_percentile"] >= "0.900000000000"
        ]
    except (SystemExit, Exception) as exc:  # noqa: BLE001 - report, never invent
        # The candidate lane is one input, not the digest: report the gap so the
        # panel can say INSUFFICIENT_EVIDENCE for candidates and still run.
        return {
            "status": "UNAVAILABLE",
            "error": str(exc) or type(exc).__name__,
            "shortlist": [],
            "shortlist_size": 0,
        }
    return {
        "status": "READY",
        "universe_refs": refs,
        "ranked": len(ranked),
        "after_gates": len(gates["keep"]),
        "gated_out": {name: sorted(values) for name, values in gates["dropped"].items()},
        "shortlist": shortlist[:20],
        "shortlist_size": len(shortlist),
        "gates": {
            "exclude_risk_warning_names": True,
            "minimum_listed_days": planner.ENTRY_MINIMUM_LISTED_DAYS,
            "minimum_daily_turnover_wan": planner.ENTRY_MINIMUM_TURNOVER_WAN,
        },
    }


def _risk(session: str) -> list[dict]:
    from quant_investor.paper.planning import position_views, session_dates
    from quant_investor.paper.store import PaperStore

    account = PaperStore(WORKSPACE).load_account(ACCOUNT_ID)
    views = position_views(
        workspace=WORKSPACE,
        account=account,
        as_of=session,
        dates=session_dates(WORKSPACE, as_of=session),
    )
    return [
        {
            "symbol": view["symbol"],
            "name": view["name"],
            "close": view["close"],
            "nav_weight": view["nav_weight"],
            "hard_stop": view["hard_stop"],
            "giveback_ratio": view["giveback_ratio"],
            "peak_price": view["peak_price"],
            "moving_stop": view["review_price"],
            "moving_reduce": view["reduce_price"],
            "trailing_trigger": view["trailing_trigger"],
            "owner_stop_trigger": view["owner_stop_trigger"],
            "blockers": view["blockers"],
        }
        for view in views
    ]


def _concentration(risk_rows: list[dict]) -> dict:
    """The paper policy's own concentration rules, checked against the account.

    Every limit is read from the approved policy file — nothing here invents a
    threshold. `observed` values are derived ratios: no symbols, no money.
    """

    from decimal import Decimal

    from quant_investor.paper import contracts

    raw = (WORKSPACE / contracts.POLICY_RELATIVE_PATH).read_bytes()
    entry = json.loads(raw)["entry_policy"]
    weights = [Decimal(str(row["nav_weight"])) for row in risk_rows]
    top1 = max(weights) if weights else Decimal("0")
    cash_fraction = Decimal("1") - sum(weights, Decimal("0"))
    headroom = Decimal("0.000001")
    checks = [
        {
            "rule_id": "maximum_holdings",
            "limit": entry["maximum_holdings"],
            "observed": len(weights),
            "result": "PASS" if len(weights) <= int(entry["maximum_holdings"]) else "FAIL",
            "applies_to": "PORTFOLIO_INVARIANT",
        },
        {
            "rule_id": "target_weight_per_holding",
            "limit": entry["target_weight_per_holding"],
            "observed": f"{top1:.6f}",
            # Not PASS/FAIL: the cap binds entries, and a position above it from
            # appreciation is not a breach — but it is surfaced, not hidden.
            "result": (
                "PASS"
                if top1 <= Decimal(entry["target_weight_per_holding"]) + headroom
                else "ABOVE_ENTRY_CAP"
            ),
            "applies_to": "ENTRY_SIZING_CAP",
            "note": "仅约束买入规模；持仓因上涨超过该比例本身不是减仓指令，也不是组合级上限",
        },
        {
            "rule_id": "minimum_cash_fraction",
            "limit": entry["minimum_cash_fraction"],
            "observed": f"{cash_fraction:.6f}",
            "result": (
                "PASS" if cash_fraction >= Decimal(entry["minimum_cash_fraction"]) else "FAIL"
            ),
            "applies_to": "PORTFOLIO_INVARIANT",
        },
    ]
    return {
        "policy_ref": {
            "path": contracts.POLICY_RELATIVE_PATH,
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        "position_count": len(weights),
        "checks": checks,
        "note": (
            "阈值取自获批政策本体；组合层集中度上限（单票/行业）尚无获批阈值，"
            "仅 entry sizing 上限可用"
        ),
    }


def _drawdown(session: str) -> dict:
    from quant_investor.paper.planning import account_nav_history

    return account_nav_history(workspace=WORKSPACE, account_id=ACCOUNT_ID, as_of=session)


def _seal_veto() -> dict:
    """The production chain's write-veto state, as evidence refs only."""

    sources: list[dict] = []
    state = "CLEAR"
    for scope, name in (
        ("market_maintenance", "WRITE_VETO.json"),
        ("macro_release", "MACRO_WRITE_VETO.json"),
    ):
        path = VETO_ROOT / name
        if not path.exists():
            sources.append({"scope": scope, "present": False, "state": "CLEAR", "ref": None})
            continue
        raw = path.read_bytes()
        value = json.loads(raw)
        state = "ACTIVE"
        sources.append(
            {
                "scope": scope,
                "present": True,
                "state": "ACTIVE",
                "ref": {
                    "path": path.relative_to(WORKSPACE).as_posix(),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                },
                "blockers": value.get("blockers", []),
                "target_date": value.get("target_date"),
                "created_at": value.get("created_at"),
            }
        )
    attempts = sorted(VETO_ROOT.glob("attempts/*/attempt.json"))
    if attempts:
        newest = attempts[-1]
        raw = newest.read_bytes()
        value = json.loads(raw)
        sources.append(
            {
                "scope": "latest_maintenance_attempt",
                "present": True,
                "state": "BLOCKED" if value.get("blockers") else "READY",
                "ref": {
                    "path": newest.relative_to(WORKSPACE).as_posix(),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                },
                "status": value.get("status"),
                "maintenance_status": value.get("maintenance_status"),
                "workflow_status": value.get("workflow_status"),
                "blockers": value.get("blockers", []),
                "macro_status": value.get("macro_status"),
                "factor_loop": value.get("factor_loop"),
            }
        )
    return {
        "state": state,
        "sources": sources,
        "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": (
            "ACTIVE 表示生产链存在未解除的写入否决；panel 只读引用，"
            "不解锁、不绕过，也不拿它代替交易判断"
        ),
    }


def _orders(session: str) -> dict:
    """Outstanding orders: everything planned that the account has not applied.

    On the evening of `session` the plan for the next session has just been
    written, while an order that pended on an earlier session still carries;
    both are what the panel reviews. Applied intents are history, not proposals.
    """

    from quant_investor.paper.planning import outstanding_orders

    applied = set(_account().get("applied_source_intents") or [])
    grouped: dict[str, dict] = {}
    for item in outstanding_orders(workspace=WORKSPACE, applied=applied):
        key = item["plans"].as_posix()
        row = grouped.setdefault(
            key,
            {
                "plans_ref": _ref(item["plans"]),
                "eligible_from_trade_date": item["due"],
                "orders": [],
            },
        )
        row["orders"].append(item["order"])
    return {"evaluated_session": session, "plans": list(grouped.values())}


def _evidence(session: str) -> dict:
    lanes = {
        "strict_closes": WORKSPACE / "data/parquet/cn/_latest.json",
        "theme_universe": WORKSPACE
        / "data/private/paper_evidence"
        / session
        / "paper-technology-universe.v1.json",
        "paper_limits": WORKSPACE
        / "data/private/paper_evidence"
        / session
        / "paper-price-limit-evidence.v1.json",
        "risk_calculator": WORKSPACE
        / "data/private/paper_evidence"
        / session
        / "risk-monitor.json",
        "research_pool": WORKSPACE
        / "results/intelligence/research_pool/aggressive_tech_manufacturing"
        / f"{session[:4]}-{session[4:6]}-{session[6:]}"
        / "factor_research_rank.json",
        "theme_replay": None,
        "industry": None,
        "fundamental": None,
        "macro": None,
        "corporate_action_recon": None,
    }
    replays = sorted(
        (WORKSPACE / "data/private/intelligence_sources/theme/replays").glob(
            f"theme-replay-{session}-*.json"
        )
    )
    if replays:
        lanes["theme_replay"] = replays[-1]
    present = {}
    missing = []
    for lane, path in lanes.items():
        if path is not None and Path(path).exists():
            present[lane] = _ref(Path(path))
        else:
            missing.append(lane)
    return {
        "present": present,
        "missing": sorted(missing),
        "note": (
            "缺少的 lane 必须报告 INSUFFICIENT_EVIDENCE，不得用历史值、推测或常识补齐。"
            "industry/fundamental/macro/corporate_action_recon 属于尚未接通的研究半链。"
        ),
    }


def build(session: str) -> dict:
    risk_rows = _risk(session)
    digest = {
        "schema_version": "cn-decision-digest.v1",
        "trade_date": session,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "account": _account(),
        "candidates": _candidates(session),
        "risk": risk_rows,
        "concentration": _concentration(risk_rows),
        "drawdown": _drawdown(session),
        "seal_veto": _seal_veto(),
        "orders": _orders(session),
        "evidence": _evidence(session),
        "authority": {
            "broker": False,
            "real_order": False,
            "actual_holdings_mutation": False,
            "research_only": True,
            "note": "本 digest 是只读研究输入；任何 agent 的输出都不是交易指令。",
        },
    }
    raw = json.dumps(digest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    digest["content_sha256"] = hashlib.sha256(raw).hexdigest()
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trade-date", help="YYYYMMDD; default is the latest closed session")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    session = args.trade_date or _session()
    digest = build(session)
    if args.write:
        path = GENERATED / session / "cn-decision-digest.v1.json"
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.write_bytes(json.dumps(digest, ensure_ascii=False, indent=1, sort_keys=True).encode())
        path.chmod(0o600)
        print(
            json.dumps(
                {
                    "status": "WRITTEN",
                    "path": str(path.relative_to(WORKSPACE)),
                    "content_sha256": digest["content_sha256"],
                },
                ensure_ascii=False,
            )
        )
    else:
        print(json.dumps(digest, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
