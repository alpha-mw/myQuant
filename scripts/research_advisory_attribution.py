#!/usr/bin/env python3
"""Did the advice in the decision log pay off, and did the owner follow it?

Every advisory and owner action in ``results/decision_log/decision_log.jsonl``
names a date, usually a symbol and an action. Nothing joins them to what the
stock did next, so the only LLM output that reaches real money (Codex-thread
advice the owner acts on by hand) has never been measured.

For each directional, symbol-level event this script takes the next session's
open as the actionable price and measures the adjusted return to the close
5, 20 and 60 sessions later, less the equal-weight universe return over the
same window. A call "pays off" when direction times that excess is positive.
It also records whether the production LOW/W80 top-100 pool held the symbol on
the advice date, so advice can be compared with the model.

Research only: it reads the canonical bars of the current snapshot and writes
only under ``reports/research/``. The log is small and the calls are few, so
treat the aggregates as bookkeeping that accumulates, not as evidence yet.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from research_factor_topn_backtest import (  # noqa: E402
    ROOT,
    blend_w80,
    compute_components,
    listed_mask,
    listing_dates,
    load_bars,
    rank_pool,
    universe_benchmarks,
)

HORIZONS = (5, 20, 60)
SYMBOL = re.compile(r"\b([0-9]{6}\.(?:SH|SZ|BJ))\b")
LONG_ACTIONS = ("buy", "add_risk", "add")
SHORT_ACTIONS = ("sell", "local/manual sell", "reduce_risk", "clear_risk", "reduce", "exit")
HOLD_ACTIONS = ("hold", "watch")


@dataclass(frozen=True)
class Call:
    event_id: str
    event_type: str
    source: str
    trade_date: str
    symbol: str
    action: str
    direction: int


def classify_action(action: str | None) -> int | None:
    """+1 for a long call, -1 for a risk-off call, 0 for hold, None when unclassified."""

    text = (action or "").strip().lower()
    if not text:
        return None
    for prefix in SHORT_ACTIONS:
        if text.startswith(prefix):
            return -1
    for prefix in LONG_ACTIONS:
        if text.startswith(prefix):
            return 1
    for prefix in HOLD_ACTIONS:
        if text.startswith(prefix):
            return 0
    return None


def extract_calls(rows: list[dict]) -> tuple[list[Call], list[dict]]:
    """Directional symbol-level calls, de-duplicated; everything else is listed as skipped."""

    calls: dict[tuple[str, str, str, int], Call] = {}
    skipped: list[dict] = []
    for row in rows:
        if row.get("event_type") not in {"advisory", "human_action"}:
            skipped.append({"event_id": row.get("event_id"), "reason": "not_an_advice_or_action"})
            continue
        direction = classify_action(row.get("action"))
        symbols = SYMBOL.findall(f"{row.get('symbol') or ''} {row.get('action') or ''}")
        if direction is None or direction == 0 or not symbols:
            reason = "hold_or_watch" if direction == 0 else "no_direction_or_symbol"
            skipped.append({"event_id": row.get("event_id"), "reason": reason})
            continue
        source = str(row.get("answer_source") or row.get("channel") or "")
        for symbol in dict.fromkeys(symbols):
            key = (row["trade_date"], symbol, row["event_type"], direction)
            calls.setdefault(
                key,
                Call(
                    event_id=str(row["event_id"]),
                    event_type=str(row["event_type"]),
                    source=source,
                    trade_date=str(row["trade_date"]).replace("-", ""),
                    symbol=symbol,
                    action=str(row["action"]),
                    direction=direction,
                ),
            )
    return list(calls.values()), skipped


def forward_returns(
    bars: pd.DataFrame, benchmark: pd.DataFrame, call: Call
) -> dict[str, float | str | None]:
    """Next-open entry, close-to-close exits, less the equal-weight universe."""

    history = bars[bars["ts_code"] == call.symbol].reset_index(drop=True)
    later = history.index[history["trade_date"] > call.trade_date]
    if len(later) == 0:
        return {"entry_date": None}
    entry = int(later[0])
    entry_price = history.at[entry, "open"] * history.at[entry, "adj_factor"]
    entry_date = history.at[entry, "trade_date"]
    cumulative = (1.0 + benchmark["equal_weight"]).cumprod()
    result: dict[str, float | str | None] = {"entry_date": entry_date}
    for horizon in HORIZONS:
        exit_row = entry + horizon - 1
        if exit_row >= len(history):
            result[f"excess_{horizon}"] = None
            continue
        exit_date = history.at[exit_row, "trade_date"]
        stock = history.at[exit_row, "adj_close"] / entry_price - 1.0
        # The benchmark is close-to-close, so start it from the session before entry.
        universe = (
            cumulative.loc[exit_date]
            / cumulative.loc[entry_date]
            * (1.0 + benchmark.at[entry_date, "equal_weight"])
            - 1.0
        )
        result[f"stock_{horizon}"] = float(stock)
        result[f"excess_{horizon}"] = float(stock - universe)
    return result


def pool_membership(bars: pd.DataFrame, dates: list[str]) -> dict[str, set[str]]:
    """The production top-100 pool at the close of each advice date (or the prior session)."""

    sessions = sorted(bars["trade_date"].unique())
    eligible = bars[bars["listed"] & (bars["total_mv"] > 0.0)]
    pools: dict[str, set[str]] = {}
    for day in dates:
        session = max((value for value in sessions if value <= day), default=None)
        if session is None:
            pools[day] = set()
            continue
        frame = eligible[eligible["trade_date"] == session].set_index("ts_code")
        pools[day] = set(rank_pool(frame["low_dollar"], blend_w80(frame), top_n=100))
    return pools


def summarise(table: pd.DataFrame) -> dict:
    summary: dict = {"calls": int(len(table))}
    for group_name, group in (("all", table), *table.groupby("event_type")):
        block: dict = {"calls": int(len(group))}
        for horizon in HORIZONS:
            column = group[f"excess_{horizon}"].dropna()
            signed = column * group.loc[column.index, "direction"]
            block[f"h{horizon}"] = {
                "n": int(len(signed)),
                "mean_signed_excess": float(signed.mean()) if len(signed) else None,
                "hit_rate": float((signed > 0).mean()) if len(signed) else None,
            }
        summary[str(group_name)] = block
    advisories = table[table["event_type"] == "advisory"]
    actions = table[table["event_type"] == "human_action"]
    followed = sum(
        1
        for _, row in advisories.iterrows()
        if (
            (actions["symbol"] == row["symbol"])
            & (actions["direction"] == row["direction"])
            & (actions["trade_date"] >= row["trade_date"])
        ).any()
    )
    summary["advisories_followed_by_an_owner_action"] = followed
    summary["advisories_in_model_pool"] = int(advisories["in_model_pool"].sum())
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    parser.add_argument(
        "--decision-log", type=Path, default=Path("results/decision_log/decision_log.jsonl")
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    log_path = args.decision_log
    if not log_path.is_absolute():
        log_path = args.workspace_root / log_path
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    calls, skipped = extract_calls(rows)
    if not calls:
        print(json.dumps({"status": "NO_CALLS", "skipped": len(skipped)}))
        return 0

    first = min(call.trade_date for call in calls)
    bars, snapshot = load_bars(
        args.workspace_root, start=f"{int(first[:4]) - 1}0101", end="99991231"
    )
    bars["listed"] = listed_mask(bars, listing_dates(args.workspace_root))
    bars = bars.join(compute_components(bars))
    sessions = sorted(bars["trade_date"].unique())
    benchmark = universe_benchmarks(bars, sessions[1:])
    pools = pool_membership(bars, sorted({call.trade_date for call in calls}))

    records = []
    for call in calls:
        records.append(
            {
                **call.__dict__,
                "in_model_pool": call.symbol in pools[call.trade_date],
                **forward_returns(bars, benchmark, call),
            }
        )
    table = pd.DataFrame(records).sort_values(["trade_date", "symbol", "event_type"])
    summary = {
        "schema": "research-advisory-attribution.v1",
        "authority": "RESEARCH_ONLY_NOT_INVESTMENT_ADVICE",
        "snapshot": snapshot,
        "decision_log_events": len(rows),
        "skipped": skipped,
        "summary": summarise(table),
        "caveat": "A handful of calls; hit rates here are bookkeeping, not evidence.",
    }
    output = args.output_dir or (
        args.workspace_root / "reports/research/advisory_attribution" / snapshot["snapshot_id"]
    )
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "calls.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "COMPLETED", "output_dir": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
