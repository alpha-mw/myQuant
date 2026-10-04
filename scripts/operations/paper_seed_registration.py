#!/usr/bin/env python3
"""Build a Paper account genesis registration from the active sealed record.

Read-only with respect to production state: the registration is written to a
private path (default ``data/private/paper_account_seed/``) and nothing else
changes. The registration itself is only consumed later by
``python -m quant_investor paper account-register --allow-write``, which must run
from the installed release.

Per-position ``realized_pnl`` and ``cumulative_fees`` start at zero: the Paper
account is a fresh genesis anchored to the sealed record through
``genesis_source_ref``, not a continuation of the manual ledger's history.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

WORKSPACE = Path("/Users/maxwell/mySpace/myQuant")
RECORD_ROOT = WORKSPACE / "results/strategy_records/CN/aggressive_tech_manufacturing"
POINTER = RECORD_ROOT / "_record_store/current.v1.json"
WRITER_ID = "cn-paper-risk-exit-writer.v1"
SEED_ROOT = WORKSPACE / "data/private/paper_account_seed"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def money(value) -> str:
    return f"{float(value):.4f}"


def compact(value: str) -> str:
    return value.replace("-", "")


def build(account_id: str) -> dict:
    import pandas as pd

    from quant_investor.paper.contracts import (
        POLICY_RELATIVE_PATH,
        POLICY_SHA256,
        seal_document,
    )

    pointer = json.loads(POINTER.read_text())
    record_dir = RECORD_ROOT / pointer["active_record_id"]
    ledger_path = record_dir / "ledger_after_manual_switch.parquet"
    frame = pd.read_parquet(ledger_path)
    summary = record_dir / "pnl_summary.csv"
    with summary.open() as handle:
        import csv as _csv

        last = list(_csv.DictReader(handle))[-1]
    valuation_date = compact(last["quote_snapshot"].split("_")[0])

    positions = []
    for row in frame.sort_values("symbol").to_dict("records"):
        entry = row.get("manual_entry_trade_date")
        acquisition = compact(str(entry)) if entry else valuation_date
        positions.append(
            {
                "symbol": row["symbol"],
                "name": row["name"],
                "shares": int(row["shares"]),
                "settled_shares": int(row["shares"]),
                "avg_cost": money(row["avg_cost"]),
                "cost_basis": money(row["cost_basis"]),
                "realized_pnl": "0.0000",
                "cumulative_fees": "0.0000",
                "acquisition_lots": [
                    {
                        "shares": int(row["shares"]),
                        "acquisition_date": acquisition,
                        "settlement_date": valuation_date,
                    }
                ],
            }
        )

    registration = seal_document(
        {
            "schema_version": "paper-account-registration.v1",
            "account_id": account_id,
            "account_type": "PAPER",
            "strategy_id": "aggressive_tech_manufacturing",
            "currency": "CNY",
            "allowed_writer_id": WRITER_ID,
            "policy_ref": {"path": POLICY_RELATIVE_PATH, "sha256": POLICY_SHA256},
            "genesis_source_ref": {
                "path": str(ledger_path.relative_to(WORKSPACE)),
                "sha256": sha256_file(ledger_path),
            },
            "initial_cash": money(last["cash_after"]),
            "initial_positions": positions,
            "all_initial_shares_settled": True,
            "broker": False,
            "real_order": False,
            "actual_holdings_mutation": False,
        }
    )
    return {
        "registration": registration,
        "seed_context": {
            "active_record_id": pointer["active_record_id"],
            "valuation_trade_date": valuation_date,
            "ledger_sha256": sha256_file(ledger_path),
            "account_total_value_cny": last["total_value_after"],
            "position_count": len(positions),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    from quant_investor.paper.contracts import validate_registration

    from quant_investor.contracts import canonical_json_bytes

    payload = build(args.account_id)
    registration = payload["registration"]
    normalized = validate_registration(registration)
    if {k: v for k, v in normalized.items() if k != "semantic_sha256"} != {
        k: v for k, v in registration.items() if k != "semantic_sha256"
    }:
        raise SystemExit("registration does not survive validation unchanged")
    encoded = canonical_json_bytes(registration)
    raw = encoded.decode()
    print(f"registration_sha256 {hashlib.sha256(encoded).hexdigest()}")
    print(f"semantic_sha256     {registration['semantic_sha256']}")
    print(json.dumps(payload["seed_context"], ensure_ascii=False, indent=1))
    if args.write:
        target = SEED_ROOT / args.account_id
        target.mkdir(parents=True, exist_ok=True)
        path = target / "paper-account-registration.v1.json"
        path.write_bytes(encoded)
        print(f"wrote {path}")
    else:
        print(raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
