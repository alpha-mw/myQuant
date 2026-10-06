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


def _number(value) -> str | None:
    import math

    if value is None:
        return None
    text = float(value)
    if math.isnan(text) or math.isinf(text):
        return None
    return f"{text:.6f}"


def _research_industry(session: str, symbols: list[str]) -> dict:
    """Per-symbol industry from the sealed PIT membership generation."""

    import pandas as pd

    pointer_path = WORKSPACE / "data/parquet/cn/reference/stock_basic_membership_latest.json"
    pointer_raw = pointer_path.read_bytes()
    pointer = json.loads(pointer_raw)
    membership = Path(pointer["canonical_path"])
    if hashlib.sha256(membership.read_bytes()).hexdigest() != pointer["canonical_sha256"]:
        raise SystemExit("PIT membership bytes differ from the pointer's sha")
    frame = pd.read_parquet(
        membership,
        columns=[
            "symbol",
            "name",
            "industry",
            "board_market",
            "effective_from",
            "effective_to",
            "membership_quality",
        ],
    )
    picked: dict[str, dict] = {}
    for row in frame.itertuples(index=False):
        start = str(row.effective_from or "").replace("-", "")
        end = str(row.effective_to or "").replace("-", "")
        if start and start > session:
            continue
        if end and end < session:
            continue
        current = picked.get(row.symbol)
        if current is None or start > str(current["effective_from"]):
            picked[row.symbol] = {
                "name": row.name,
                "industry": row.industry or None,
                "board_market": row.board_market or None,
                "effective_from": start or None,
                "membership_quality": row.membership_quality,
            }
    return {
        "status": "READY",
        "source": pointer.get("source"),
        "generation_id": pointer.get("generation_id"),
        "observed_at": pointer.get("observed_at"),
        "pointer_ref": {
            "path": pointer_path.relative_to(WORKSPACE).as_posix(),
            "sha256": hashlib.sha256(pointer_raw).hexdigest(),
        },
        "membership_ref": {
            "path": str(membership.relative_to(WORKSPACE)),
            "sha256": pointer["canonical_sha256"],
        },
        "symbols": {symbol: picked.get(symbol) for symbol in symbols},
        "note": "行业字段来自 tushare.stock_basic 的 PIT 快照（当次 observed_at），非申万分类",
    }


def _research_fundamental(session: str, symbols: list[str]) -> dict:
    """Latest PIT fundamental row per symbol, with its own cutoff and lag."""

    import pandas as pd

    from quant_investor.market.fundamental_generation import (
        load_fundamental_pointer,
        resolve_fundamental_table_path,
    )

    root = WORKSPACE / "data/parquet/cn"
    pointer = load_fundamental_pointer(root)
    if pointer is None:
        return {"status": "MISSING", "reason": "FUNDAMENTAL_POINTER_MISSING"}
    table = resolve_fundamental_table_path(root, "fundamental_daily")
    columns = [
        "ts_code",
        "trade_date",
        "end_date",
        "availability_date",
        "total_mv_rmb",
        "fin_roe",
        "fin_roa",
        "fin_debt_to_assets",
        "fin_net_profit_yoy",
        "fin_ocf_to_profit",
        "fin_fcf_to_profit",
        "fcf_to_price",
        "forecast_revision",
        "forecast_type",
        "sector",
        "size_bucket",
    ]
    frame = pd.read_parquet(table, columns=columns, filters=[("ts_code", "in", symbols)])
    if frame.empty:
        return {
            "status": "MISSING",
            "reason": "NO_ROWS_FOR_SYMBOLS",
            "generation_id": pointer.get("generation_id"),
        }
    frame["trade_date"] = frame["trade_date"].astype(str).str.replace("-", "")
    cutoff = str(frame["trade_date"].max())
    frame = frame[frame["trade_date"] <= session]
    rows: dict[str, dict] = {}
    for row in frame.sort_values("trade_date").groupby("ts_code").tail(1).itertuples(index=False):
        rows[row.ts_code] = {
            "trade_date": row.trade_date,
            "end_date": str(row.end_date).replace("-", "").split(" ")[0],
            "availability_date": str(row.availability_date).replace("-", "").split(" ")[0],
            "total_mv_yi_cny": (
                None if row.total_mv_rmb is None else f"{float(row.total_mv_rmb) / 1e8:.2f}"
            ),
            "roe": _number(row.fin_roe),
            "roa": _number(row.fin_roa),
            "debt_to_assets": _number(row.fin_debt_to_assets),
            "net_profit_yoy": _number(row.fin_net_profit_yoy),
            "ocf_to_profit": _number(row.fin_ocf_to_profit),
            "fcf_to_profit": _number(row.fin_fcf_to_profit),
            "fcf_to_price": _number(row.fcf_to_price),
            "forecast_revision": _number(row.forecast_revision),
            "forecast_type": None if row.forecast_type is None else str(row.forecast_type),
            "sector": None if row.sector is None else str(row.sector),
            "size_bucket": None if row.size_bucket is None else str(row.size_bucket),
        }
    manifest_path = table.parent / "manifest.json"
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    declared = ((manifest.get("tables") or {}).get("fundamental_daily") or {}).get("sha256")
    lag_days = (
        datetime.strptime(session, "%Y%m%d").date() - datetime.strptime(cutoff, "%Y%m%d").date()
    ).days
    statement_periods = sorted({row["end_date"] for row in rows.values() if row["end_date"]})
    announcement_dates = sorted(
        {row["availability_date"] for row in rows.values() if row["availability_date"]}
    )
    return {
        "status": "READY" if lag_days <= 0 else "STALE",
        "cutoff_trade_date": cutoff,
        "lag_days": lag_days,
        "statement_period_end": statement_periods[-1] if statement_periods else None,
        "announcement_available_from": (announcement_dates[0] if announcement_dates else None),
        "generation_id": pointer.get("generation_id"),
        "pointer_ref": {
            "path": "data/parquet/cn/_fundamental_latest.json",
            "sha256": hashlib.sha256((root / "_fundamental_latest.json").read_bytes()).hexdigest(),
        },
        "manifest_ref": {
            "path": str(manifest_path.relative_to(WORKSPACE)),
            "sha256": hashlib.sha256(manifest_raw).hexdigest(),
        },
        "table_ref": {
            "path": str(table.relative_to(WORKSPACE)),
            "sha256": declared,
        },
        "symbols": rows,
        "note": (
            "报表口径为最近已公告报告期（statement_period_end / announcement_available_from，"
            "PIT 自公告日生效）；日频行（市值、fcf_to_price 等）截止 cutoff_trade_date，"
            "滞后 lag_days 个自然日。估值类结论要么用 digest 的当日收盘自行重算，"
            "要么明确标注该滞后；不得把日频行当作当日证据"
        ),
    }


def _research_macro() -> dict:
    """The macro observation store as it stands: observer-only, not applied."""

    from quant_investor.macro.store import load_observations

    root = WORKSPACE / "data/parquet/cn/macro_observations"
    pointer_path = root / "_latest.json"
    pointer_raw = pointer_path.read_bytes()
    pointer = json.loads(pointer_raw)
    rows, _meta = load_observations(root)
    observations = [
        {
            "indicator_id": row.get("indicator_id"),
            "period_end": row.get("period_end"),
            "release_at": row.get("release_at"),
            "available_at": row.get("available_at"),
            "value": None if row.get("value") is None else str(row["value"]),
            "unit": row.get("unit"),
            "quality_status": row.get("quality_status"),
        }
        for row in rows
    ]
    calendar_path = WORKSPACE / "data/parquet/cn/macro_release_calendar/_latest.json"
    calendar_raw = calendar_path.read_bytes()
    return {
        "status": "OBSERVER_ONLY" if not pointer.get("production_eligible") else "READY",
        "generation_id": pointer.get("generation_id"),
        "production_eligible": pointer.get("production_eligible"),
        "applied": pointer.get("applied"),
        "observer_only": pointer.get("observer_only"),
        "pointer_ref": {
            "path": pointer_path.relative_to(WORKSPACE).as_posix(),
            "sha256": hashlib.sha256(pointer_raw).hexdigest(),
        },
        "release_calendar_ref": {
            "path": calendar_path.relative_to(WORKSPACE).as_posix(),
            "sha256": hashlib.sha256(calendar_raw).hexdigest(),
        },
        "observations": observations,
        "note": (
            "observer-only 且未 applied：只能作为背景，不构成可执行证据；"
            "宏观写入端另有未解除 veto（见 seal_veto）"
        ),
    }


def _research_market_multiples() -> dict:
    """The PE/PB/市值 side table, reported by its true currency.

    It is not part of the sealed snapshot (no pointer), so this reads the
    Parquet row-group statistics instead of claiming a sealed ref: the point is
    only to tell the lanes whether today's multiples exist at all.
    """

    import pyarrow.parquet as pq

    path = WORKSPACE / "data/parquet/cn/daily_basic/part.parquet"
    if not path.exists():
        return {"status": "MISSING", "reason": "DAILY_BASIC_TABLE_ABSENT"}
    parquet = pq.ParquetFile(path)
    index = parquet.schema_arrow.get_field_index("trade_date")
    maxima = []
    for group in range(parquet.metadata.num_row_groups):
        statistics = parquet.metadata.row_group(group).column(index).statistics
        if statistics is not None and statistics.max is not None:
            maxima.append(str(statistics.max).replace("-", "")[:8])
    latest = max(maxima) if maxima else None
    return {
        "status": "STALE" if latest else "UNKNOWN",
        "latest_trade_date": latest,
        "columns": ["total_mv", "circ_mv", "pe", "pb", "turnover_rate"],
        "row_count": parquet.metadata.num_rows,
        "path": path.relative_to(WORKSPACE).as_posix(),
        "sealed": False,
        "note": (
            "非封存旁表（无 pointer/sha 绑定），仅以 row-group 统计报告其截止日期；"
            "晚于该日期的 PE/PB/市值在本工作区没有证据，不得从别处补"
        ),
    }


def _research(session: str, symbols: list[str]) -> dict:
    return {
        "industry": _research_industry(session, symbols),
        "fundamental": _research_fundamental(session, symbols),
        "market_multiples": _research_market_multiples(),
        "macro": _research_macro(),
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
        "industry": WORKSPACE / "data/parquet/cn/reference/stock_basic_membership_latest.json",
        "fundamental": None,
        "macro": None,
        "corporate_action_recon": WORKSPACE
        / "data/private/paper_evidence"
        / session
        / "paper-corporate-action-reconciliation.v1.json",
    }
    replays = sorted(
        (WORKSPACE / "data/private/intelligence_sources/theme/replays").glob(
            f"theme-replay-{session}-*.json"
        )
    )
    if replays:
        lanes["theme_replay"] = replays[-1]
    stale = {
        "fundamental": (
            "存在但日频行 cutoff 早于本次 session（见 research.fundamental.cutoff_trade_date/"
            "lag_days）：已公告报表口径可用，当日估值/市值论断不可用"
        ),
        "macro": (
            "observer-only 且未 applied（见 research.macro）；只能作背景，"
            "不构成可执行证据，且宏观写入端仍有未解除 veto"
        ),
    }
    present = {}
    missing = []
    for lane, path in lanes.items():
        if path is not None and Path(path).exists():
            present[lane] = _ref(Path(path))
        elif lane not in stale:
            missing.append(lane)
    return {
        "present": present,
        "stale": stale,
        "missing": sorted(missing),
        "note": (
            "present 可引用；stale 的 lane 必须标注滞后并按角色判断可用范围；"
            "missing 的 lane 必须报告 INSUFFICIENT_EVIDENCE，不得用历史值、推测或常识补齐。"
        ),
    }


def build(session: str) -> dict:
    risk_rows = _risk(session)
    account = _account()
    candidates = _candidates(session)
    research_symbols = sorted(
        {row["symbol"] for row in account.get("positions", [])}
        | {row["symbol"] for row in candidates.get("shortlist", [])}
    )
    digest = {
        "schema_version": "cn-decision-digest.v1",
        "trade_date": session,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "account": account,
        "candidates": candidates,
        "risk": risk_rows,
        "concentration": _concentration(risk_rows),
        "drawdown": _drawdown(session),
        "seal_veto": _seal_veto(),
        "research": _research(session, research_symbols),
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
