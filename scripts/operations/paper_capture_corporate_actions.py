#!/usr/bin/env python3
"""Reconcile the held symbols' corporate actions for one session.

The risk calculator refuses to compute thresholds when a position's window
contains more than one `adj_factor`, because a cash dividend or a share change
moves both the price and the calibrated stop. That review needs the provider's
own record, which the workspace does not hold (its `dividend` table has four
rows), so this captures it:

- one `dividend` request per affected symbol (official endpoint);
- the local `adj_factor` series for the same window, from the strict snapshot;
- a sealed reconciliation naming the ex-date, the factor before/after, the
  provider's cash/share terms, and the **re-anchor**: the window restarts at the
  last ex-date, so the sealed calculator can evaluate the current stop against
  post-action prices instead of refusing the whole lane.

Everything lands owner-only under `data/private/paper_evidence/<date>/`.
A symbol whose adjustment the provider does not explain is left unresolved
rather than assumed away.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat

WORKSPACE = Path("/Users/maxwell/mySpace/myQuant")
EVIDENCE_ROOT = WORKSPACE / "data/private/paper_evidence"
STOP_POLICY = (
    WORKSPACE
    / "results/policies/risk/aggressive_tech_manufacturing/initial-risk-stop.v1"
    / "owner-stop-policy-20260828-v1.json"
)
TRAILING_POLICY = (
    WORKSPACE
    / "results/policies/risk/aggressive_tech_manufacturing/trailing-anchor.v1"
    / "owner-trailing-anchor-policy-20260901-v1.json"
)
DAY = re.compile(r"^[0-9]{8}$")
FIELDS = (
    "ts_code",
    "end_date",
    "ann_date",
    "div_proc",
    "stk_div",
    "stk_bo_rate",
    "stk_co_rate",
    "cash_div",
    "cash_div_tax",
    "record_date",
    "ex_date",
    "pay_date",
    "div_listdate",
    "imp_ann_date",
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load_token() -> None:
    if os.environ.get("TUSHARE_TOKEN"):
        return
    for line in (WORKSPACE / ".env").read_text().splitlines():
        if line.startswith("TUSHARE_TOKEN="):
            os.environ["TUSHARE_TOKEN"] = line.split("=", 1)[1].strip()
            return
    raise SystemExit("TUSHARE_TOKEN missing")


def _write_owner_only(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_bytes(raw)
    path.chmod(0o600)
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise SystemExit(f"{path} is not owner-only")


def _adj_factor_series(symbol: str, trade_date: str) -> list[tuple[str, str]]:
    import pandas as pd

    pointer = json.loads((WORKSPACE / "data/parquet/cn/_latest.json").read_text())
    table = WORKSPACE / "data/parquet/cn/_snapshots" / pointer["snapshot_id"] / "table"
    frames = [
        pd.read_parquet(path) for path in sorted((table / "bars").glob("year=*/month=*/*.parquet"))
    ]
    frame = pd.concat(frames)
    rows = frame[(frame.ts_code == symbol) & (frame.trade_date <= trade_date)].sort_values(
        "trade_date"
    )
    return [
        (row.trade_date, f"{float(row.adj_factor):.10f}") for row in rows.itertuples(index=False)
    ]


def _changed_windows(series: list[tuple[str, str]]) -> list[dict]:
    changes = []
    for index in range(1, len(series)):
        if series[index][1] != series[index - 1][1]:
            changes.append(
                {
                    "effective_trade_date": series[index][0],
                    "adj_factor_before": series[index - 1][1],
                    "adj_factor_after": series[index][1],
                }
            )
    return changes


def affected_symbols(trade_date: str) -> dict[str, list[dict]]:
    """Held symbols whose window holds more than one adjustment factor."""

    from quant_investor.paper.store import PaperStore

    store = PaperStore(WORKSPACE)
    account_ids = store.account_ids()
    result: dict[str, list[dict]] = {}
    if not account_ids:
        return result
    loaded = store.load_account(account_ids[0])
    stops = json.loads(STOP_POLICY.read_text())
    stop_from = str(stops["effective_from"])[:10].replace("-", "")
    anchors = {
        row["symbol"]: str(row["tracking_start_date"])
        for row in json.loads(TRAILING_POLICY.read_text()).get("anchors", [])
    }
    for row in loaded["ledger"]:
        if int(row["shares"]) <= 0:
            continue
        symbol = row["symbol"]
        # The same window the sealed calculator uses: its anchor (or the session)
        # bounded by the owner-stop policy's effective date.
        start = min(anchors.get(symbol, trade_date), stop_from)
        series = _adj_factor_series(symbol, trade_date)
        window = [(day, factor) for day, factor in series if day >= start]
        changes = _changed_windows(window)
        if changes:
            result[symbol] = changes
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if DAY.fullmatch(args.trade_date) is None:
        raise SystemExit("--trade-date must be YYYYMMDD")

    from quant_investor.contracts import canonical_json_bytes
    from quant_investor.market.tushare_transport import (
        OfficialTushareHttpsClient,
        replay_tushare_response_bytes,
    )

    affected = affected_symbols(args.trade_date)
    if not affected:
        print(
            json.dumps(
                {"status": "NO_ACTION", "reason": "NO_ADJUSTMENT_CHANGE"}, ensure_ascii=False
            )
        )
        return 0

    _load_token()
    client = OfficialTushareHttpsClient(strict_decimal_decode=True)
    day_root = EVIDENCE_ROOT / args.trade_date
    reconciled: dict[str, dict] = {}
    unresolved: list[str] = []
    raw_by_symbol: dict[str, bytes] = {}
    requests: list[dict] = []
    for symbol, changes in sorted(affected.items()):
        response = client.request(
            api_name="dividend", params={"ts_code": symbol}, expected_fields=FIELDS
        )
        replayed = replay_tushare_response_bytes(
            response.raw_body, api_name="dividend", expected_fields=FIELDS
        )
        index = {name: position for position, name in enumerate(replayed.fields)}
        ex_dates = {
            str(row[index["ex_date"]])
            for row in replayed.rows
            if str(row[index["div_proc"]]) == "实施" and row[index["ex_date"]]
        }
        matched = [change for change in changes if change["effective_trade_date"] in ex_dates]
        if len(matched) != len(changes):
            unresolved.append(symbol)
            continue
        terms = []
        for row in replayed.rows:
            if str(row[index["div_proc"]]) != "实施" or not row[index["ex_date"]]:
                continue
            terms.append(
                {
                    "end_date": str(row[index["end_date"]]),
                    "ex_date": str(row[index["ex_date"]]),
                    "record_date": (
                        None
                        if row[index["record_date"]] is None
                        else str(row[index["record_date"]])
                    ),
                    "cash_div_tax_per_share": f"{float(row[index['cash_div_tax']]):.4f}",
                    "stk_div_per_share": f"{float(row[index['stk_div']]):.4f}",
                }
            )
        re_anchor = max(change["effective_trade_date"] for change in matched)
        reconciled[symbol] = {
            "changes": matched,
            "dividend_terms": sorted(terms, key=lambda item: item["ex_date"]),
            "re_anchor_trade_date": re_anchor,
            "shares_unchanged": all(float(term["stk_div_per_share"]) == 0.0 for term in terms),
        }
        raw_by_symbol[symbol] = response.raw_body
        requests.append(
            {
                "symbol": symbol,
                "request_id": replayed.request_id,
                "rows": len(replayed.rows),
                "raw_sha256": _sha(response.raw_body),
            }
        )

    if not reconciled:
        print(json.dumps({"status": "UNRESOLVED", "symbols": unresolved}, ensure_ascii=False))
        return 0

    evidence = canonical_json_bytes(
        {
            "schema_version": "paper-corporate-action-reconciliation.v1",
            "trade_date": args.trade_date,
            "source": "tushare.dividend + strict adj_factor",
            "symbols": reconciled,
            "unresolved_symbols": unresolved,
            "effect": (
                "阈值窗口自最后一个除权除息日重新锚定，使封存计算器以除权后价格"
                "评估现行止损；不重新推导历史阈值，也不改动 owner 政策"
            ),
            "authority": {"broker": False, "real_order": False, "research_only": True},
        }
    )
    capture = canonical_json_bytes(
        {
            "schema_version": "paper-corporate-action-capture.v1",
            "trade_date": args.trade_date,
            "requests": requests,
            "evidence_sha256": _sha(evidence),
            "completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "broker": False,
            "real_order": False,
        }
    )
    result = {
        "trade_date": args.trade_date,
        "reconciled": {
            symbol: entry["re_anchor_trade_date"] for symbol, entry in sorted(reconciled.items())
        },
        "unresolved": unresolved,
        "evidence_path": str(
            (day_root / "paper-corporate-action-reconciliation.v1.json").relative_to(WORKSPACE)
        ),
        "evidence_sha256": _sha(evidence),
        "write": bool(args.write),
    }
    if args.write:
        for symbol, raw in raw_by_symbol.items():
            _write_owner_only(
                day_root / "corporate-actions" / f"{symbol.replace('.', '-')}.json", raw
            )
        _write_owner_only(day_root / "corporate-actions" / "capture.json", capture)
        _write_owner_only(day_root / "paper-corporate-action-reconciliation.v1.json", evidence)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
