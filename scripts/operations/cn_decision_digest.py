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
SHADOW_ROOT = WORKSPACE / "data/private/paper_shadow"
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
    return {
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


def _orders(session: str) -> dict:
    plans: list[dict] = []
    for path in sorted(SHADOW_ROOT.glob("*/*/plans.json")):
        value = json.loads(path.read_text())
        if value.get("eligible_from_trade_date") == session:
            plans.append({"plans_ref": _ref(path), "orders": value["orders"]})
    return {"due_for": session, "plans": plans}


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
    digest = {
        "schema_version": "cn-decision-digest.v1",
        "trade_date": session,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "account": _account(),
        "candidates": _candidates(session),
        "risk": _risk(session),
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
