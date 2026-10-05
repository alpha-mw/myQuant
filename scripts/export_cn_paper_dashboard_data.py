#!/usr/bin/env python3
"""Export the Paper account's mirror bundle for the internal dashboard.

Reads the registered Paper account and its immutable records, then writes one
self-contained bundle the internal page renders:

```text
portfolio_dashboard/private/generated/cn_paper_dashboard.v1.json
portfolio_dashboard/private/generated/cn_paper_dashboard.v1.js
```

This is an **internal-only** artifact. The public site builder copies a fixed
allow-list of files (`public.html`, `styles.css`, `app.js`, three `js/` modules
and the redacted v1 bundle), so nothing here reaches the public view.

The account is the authority: holdings, cash and every fill come from
`results/paper/accounts/<id>/`, valued at the published strict session close. A
missing or unreadable account fails the export rather than publishing a stale or
partial mirror.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

WORKSPACE = Path("/Users/maxwell/mySpace/myQuant")
GENERATED = WORKSPACE / "portfolio_dashboard/private/generated"
ACCOUNT_ID = "aggressive-tech-manufacturing-paper-v1"
SNAPSHOT_POINTER = WORKSPACE / "data/parquet/cn/_latest.json"
SCHEMA_VERSION = "cn-paper-dashboard.v1"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def strict_closes(trade_date: str) -> dict[str, str]:
    """Session closes from the published snapshot; the only valuation source."""

    import pandas as pd

    pointer = _read(SNAPSHOT_POINTER)
    table = WORKSPACE / "data/parquet/cn/_snapshots" / pointer["snapshot_id"] / "table"
    files = sorted(
        (table / "bars" / f"year={trade_date[:4]}" / f"month={trade_date[4:6]}").glob("*.parquet")
    )
    if not files:
        raise SystemExit(f"no strict bars for {trade_date}")
    frame = pd.concat([pd.read_parquet(path) for path in files])
    rows = frame[frame.trade_date == trade_date]
    return {row.ts_code: f"{float(row.close):.4f}" for row in rows.itertuples(index=False)}


def account_records(account_root: Path) -> list[dict[str, Any]]:
    """Every applied transaction, oldest first, from the immutable records."""

    history: list[dict[str, Any]] = []
    for record in sorted(account_root.glob("records/[0-9]*")):
        fill_path = record / "fills.v1.json"
        state_path = record / "account_state_after.v1.json"
        intent_path = record / "intents.v1.json"
        if not (fill_path.exists() and state_path.exists()):
            continue
        fill = _read(fill_path)
        state = _read(state_path)
        intent = _read(intent_path) if intent_path.exists() else {}
        history.append(
            {
                "record": record.name,
                "fill": fill,
                "state": state,
                "intent": intent,
            }
        )
    return history


def _action_of(intent: dict[str, Any], fill: dict[str, Any]) -> str:
    if intent.get("price_basis"):
        return "OWNER_EXIT" if intent.get("action") == "EXIT_100" else "OWNER_DIRECTED"
    if fill.get("side") == "BUY":
        return "ENTRY"
    return intent.get("action", fill.get("side", ""))


def build() -> dict[str, Any]:
    from quant_investor.paper.store import PaperStore

    store = PaperStore(WORKSPACE)
    if ACCOUNT_ID not in store.account_ids():
        raise SystemExit("paper account is not registered")
    loaded = store.load_account(ACCOUNT_ID)
    account_root = store.account_root(ACCOUNT_ID)
    state = loaded["state"]

    trades: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    realized = 0.0
    fees_paid = 0.0
    for entry in account_records(account_root):
        fill = entry["fill"]
        after = entry["state"]
        realized += float(fill["realized_pnl_delta"])
        fees_paid += float(fill["total_fees"])
        trades.append(
            {
                "record": entry["record"],
                "trade_date": fill["trade_date"],
                "symbol": fill["symbol"],
                "side": fill["side"],
                "action": _action_of(entry["intent"], fill),
                "shares": fill["shares"],
                "price": fill["simulated_price"],
                "gross": fill.get("gross_proceeds") or fill.get("gross_cost"),
                "total_fees": fill["total_fees"],
                "stamp_duty": fill["stamp_duty"],
                "realized_pnl_delta": fill["realized_pnl_delta"],
                "owner_declared_price": bool(fill.get("owner_declared_price")),
                "cash_after": after["cash"],
            }
        )
        curve.append(
            {
                "trade_date": fill["trade_date"],
                "sequence": after.get("sequence"),
                "cash": after["cash"],
                "realized_pnl": f"{realized:.4f}",
                "cumulative_fees": f"{fees_paid:.4f}",
            }
        )

    latest = _read(SNAPSHOT_POINTER)["latest_available_trade_date"]
    latest = str(latest).replace("-", "")
    closes = strict_closes(latest)
    positions = []
    market_value = 0.0
    for row in loaded["ledger"]:
        shares = int(row["shares"])
        if shares <= 0:
            continue
        close = closes.get(row["symbol"])
        if close is None:
            raise SystemExit(f"no strict close for {row['symbol']} on {latest}")
        value = shares * float(close)
        market_value += value
        positions.append(
            {
                "symbol": row["symbol"],
                "name": row["name"],
                "shares": shares,
                "settled_shares": int(row["settled_shares"]),
                "avg_cost": f"{float(row['avg_cost']):.4f}",
                "cost_basis": f"{float(row['cost_basis']):.4f}",
                "close": close,
                "market_value": f"{value:.4f}",
                "unrealized_pnl": f"{value - float(row['cost_basis']):.4f}",
                "realized_pnl": f"{float(row['realized_pnl']):.4f}",
                "cumulative_fees": f"{float(row['cumulative_fees']):.4f}",
            }
        )
    cash = float(state["cash"])
    nav = cash + market_value
    bundle: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "account_id": ACCOUNT_ID,
        "valuation_date": latest,
        "sequence": loaded["pointer"]["sequence"],
        "pointer_sha256": loaded["pointer_sha256"],
        "cash": f"{cash:.4f}",
        "market_value": f"{market_value:.4f}",
        "nav": f"{nav:.4f}",
        "initial_capital": "1000000.0000",
        "cumulative_realized_pnl": f"{float(state['realized_pnl']):.4f}",
        "cumulative_fees": f"{float(state['cumulative_fees']):.4f}",
        "positions": positions,
        "trades": trades,
        "curve": curve,
        "boundary": {
            "view": "INTERNAL_ONLY",
            "broker": False,
            "real_order": False,
            "actual_holdings_mutation": False,
            "note": "Paper 账户镜像，仅内部视图；公开站点不发布本文件。",
        },
    }
    bundle["content_sha256"] = _sha(
        json.dumps(bundle, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    )
    return bundle


def write_bundle(bundle: dict[str, Any]) -> tuple[Path, Path]:
    raw = json.dumps(bundle, ensure_ascii=False, indent=1, sort_keys=True).encode()
    js = (
        "// Generated by scripts/export_cn_paper_dashboard_data.py -- internal only.\n"
        "window.CNPaperDashboard = "
        + json.dumps(bundle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + ";\n"
    ).encode()
    GENERATED.mkdir(parents=True, exist_ok=True)
    json_path = GENERATED / "cn_paper_dashboard.v1.json"
    js_path = GENERATED / "cn_paper_dashboard.v1.js"
    json_path.write_bytes(raw)
    js_path.write_bytes(js)
    json_path.chmod(0o600)
    js_path.chmod(0o600)
    return json_path, js_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    bundle = build()
    if args.write:
        json_path, js_path = write_bundle(bundle)
        print(
            json.dumps(
                {
                    "status": "WRITTEN",
                    "json": str(json_path),
                    "js": str(js_path),
                    "content_sha256": bundle["content_sha256"],
                },
                ensure_ascii=False,
            )
        )
    else:
        print(json.dumps(bundle, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
