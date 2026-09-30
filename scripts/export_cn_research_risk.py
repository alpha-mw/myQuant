#!/usr/bin/env python3
"""Read-only shared daily/weekly risk consumer over registered Store and policies."""

from __future__ import annotations

import argparse
from datetime import date
import io
import hashlib
import json
from pathlib import Path
import sys
import stat
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
from quant_investor.market.market_data_reader import MarketDataReader  # noqa: E402
from quant_investor.macro.release_calendar import load_release_calendar  # noqa: E402
from quant_investor.strategy_records.performance import (  # noqa: E402
    assert_private_tmp,
    immutable_write,
    load_performance_history,
)
from quant_investor.strategy_records.research_risk import (  # noqa: E402
    ResearchRiskError,
    calculate_position_risk,
    decimal,
    seal,
)
from quant_investor.strategy_records.store import (  # noqa: E402
    CATALOG_SCHEMA_V3,
    load_registered_catalog,
    regular_file_sha256,
)
from quant_investor.strategy_records.event_store import load_generation as load_events  # noqa: E402

RECORD_ROOT = Path("results/strategy_records/CN/aggressive_tech_manufacturing")
TRAILING_POLICY = (
    "results/policies/risk/aggressive_tech_manufacturing/trailing-anchor.v1/"
    "owner-trailing-anchor-policy-20260901-v1.json"
)
TRAILING_SHA = "b313aa91e1f7ca1e8922b2d22f7735ceee3190675c2e8dab3955c69f0f1d342a"
STOP_POLICY = (
    "results/policies/risk/aggressive_tech_manufacturing/initial-risk-stop.v1/"
    "owner-stop-policy-20260828-v1.json"
)
STOP_SHA = "11eb7018ff6abde2d276c7b997e0e61e2cd6b170407872734bb8d407b334c178"


def build_risk_monitor(project_root: Path, *, as_of: str) -> dict[str, Any]:
    """Resolve exact sources once; never publish or consume an unregistered ledger."""
    project = project_root.resolve(strict=True)
    target = date.fromisoformat(as_of).strftime("%Y%m%d")
    refs: dict[str, str] = {}
    market_files: set[str] = set()

    def exact(relative: str | Path, expected: str | None = None) -> bytes:
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts or rel.name.lower() == "ledger.csv":
            raise ResearchRiskError("UNSAFE_SOURCE_PATH")
        path = project / rel
        if path.resolve(strict=True) != path.absolute():
            raise ResearchRiskError("SYMLINK_SOURCE_PATH")
        if rel.as_posix() in market_files:
            before = path.stat()
            if not stat.S_ISREG(before.st_mode):
                raise ResearchRiskError("MARKET_SOURCE_NOT_REGULAR")
            data = path.read_bytes()
            after = path.stat()
            fields = (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_nlink",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
            if (
                any(getattr(after, key) != getattr(before, key) for key in fields)
                or path.read_bytes() != data
            ):
                raise ResearchRiskError("MARKET_SOURCE_CHANGED_DURING_READ")
            sha = hashlib.sha256(data).hexdigest()
        else:
            sha, _ = regular_file_sha256(path, label="research risk source:" + rel.as_posix())
        if expected is not None and sha != expected:
            raise ResearchRiskError("SOURCE_SHA_MISMATCH:" + rel.as_posix())
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ResearchRiskError("SOURCE_CHANGED_DURING_READ")
        refs[rel.as_posix()] = sha
        return raw

    def ledger(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
        raw = exact(RECORD_ROOT / record["ledger_path"], record["ledger_sha256"])
        frame = pd.read_parquet(io.BytesIO(raw)).astype(object)
        frame = frame.where(pd.notna(frame), None)
        rows = frame.to_dict("records")
        if not rows or len({row["symbol"] for row in rows}) != len(rows):
            raise ResearchRiskError("LEDGER_SYMBOL_SET_INVALID")
        return {row["symbol"]: row for row in rows}

    result: dict[str, Any] = {
        "schema_id": "cn_research_risk_monitor.v1",
        "as_of": as_of,
        "status": "BLOCKED",
        "rows": [],
        "source_refs": [],
        "blockers": [],
        "executable": False,
        "investment_authority": False,
        "actions": [],
    }
    try:
        policy = json.loads(exact(TRAILING_POLICY, TRAILING_SHA))
        stops = json.loads(exact(STOP_POLICY, STOP_SHA))
        if as_of < policy["effective_from"][:10]:
            raise ResearchRiskError("POLICY_NOT_YET_EFFECTIVE")
        pointer_path = RECORD_ROOT / "_record_store/current.v1.json"
        exact(pointer_path)
        loaded = load_registered_catalog(project / RECORD_ROOT)
        if loaded is None or loaded[1].get("schema_id") != CATALOG_SCHEMA_V3:
            raise ResearchRiskError("STORE_V3_REQUIRED")
        pointer, catalog = loaded
        exact(RECORD_ROOT / pointer["catalog_path"], pointer["catalog_sha256"])
        performance = load_performance_history(
            project / RECORD_ROOT, catalog["performance_history_ref"]
        )
        for key in ("manifest", "series", "owner_declaration"):
            ref = catalog["performance_history_ref"][key]
            exact(RECORD_ROOT / ref["path"], ref["sha256"])
        confirmed_date = performance["rows"][-1]["valuation_date"]
        if confirmed_date > as_of:
            raise ResearchRiskError("HOLDINGS_AFTER_RESEARCH_DATE")
        records = {row["record_id"]: row for row in catalog["records"]}
        lineage = {row["record_id"]: row for row in catalog["lineage_index"]}
        baseline_id = policy["store_binding"]["active_record_id"]
        baseline = ledger(records[baseline_id])
        if records[baseline_id]["ledger_sha256"] != policy["store_binding"]["ledger_sha256"]:
            raise ResearchRiskError("POLICY_BASELINE_LEDGER_MISMATCH")
        current = ledger(records[pointer["active_record_id"]])
        blockers = {symbol: [] for symbol in current}
        cursor = pointer["active_record_id"]
        visited: set[str] = set()
        while cursor != baseline_id:
            if not cursor or cursor in visited or cursor not in lineage:
                raise ResearchRiskError("POLICY_BASELINE_NOT_ANCESTOR")
            visited.add(cursor)
            record = records[cursor]
            rows = ledger(record)
            manual = json.loads(
                exact(
                    RECORD_ROOT / record["manual_manifest_path"], record["manual_manifest_sha256"]
                )
            )
            for symbol in current:
                if symbol not in baseline or symbol not in rows:
                    blockers[symbol].append("POSITION_LIFECYCLE_CHANGED")
                    continue
                if any(
                    decimal(rows[symbol][field]) != decimal(baseline[symbol][field])
                    for field in ("shares", "avg_cost", "cost_basis")
                ):
                    blockers[symbol].append("POSITION_COST_OR_QUANTITY_CHANGED")
                for field in (
                    "applied_owner_declared_trades",
                    "applied_local_trades",
                    "corporate_actions",
                ):
                    if any(row.get("symbol") == symbol for row in manual.get(field, [])):
                        blockers[symbol].append("NEW_EVENT_REQUIRES_ANCHOR_REVIEW")
            cursor = lineage[cursor]["source_record_id"]
        calendar_root = Path("data/parquet/cn/macro_release_calendar")
        exact(calendar_root / "_latest.json")
        calendar = load_release_calendar(
            canonical_root=project / calendar_root,
            expected_pointer_sha256=refs[(calendar_root / "_latest.json").as_posix()],
        )
        expected_dates = [
            str(day).replace("-", "")
            for day in calendar.open_dates
            if str(day).replace("-", "") <= target
        ]
        event_root = RECORD_ROOT / "_event_store"
        exact(event_root / "current.v1.json")
        event = load_events(project / event_root)
        event_ref = event["pointer"]["generation"]
        exact(event_root / event_ref["path"], event_ref["sha256"])
        event_dates = {row["trade_date"].replace("-", "") for row in event["closures"]}
        baseline_date = lineage[baseline_id]["valuation_date"].replace("-", "")
        lifecycle_missing = [
            day for day in expected_dates if day > baseline_date and day not in event_dates
        ]
        for symbol in blockers:
            blockers[symbol].extend("LIFECYCLE_UNCONFIRMED:" + day for day in lifecycle_missing)
        reader = MarketDataReader(data_root=project / "data", mode_policy="strict")
        market = Path("data/parquet/cn/_latest.json")
        exact(market)
        snapshot = reader.snapshot()
        if not snapshot["healthy"] or target > snapshot["latest_complete_trade_date"]:
            raise ResearchRiskError("STRICT_MARKET_DATE_UNAVAILABLE")
        manifest_path = Path(snapshot["manifest_path"]).relative_to(project)
        exact(manifest_path)
        result.update(
            holdings_as_of=confirmed_date,
            holdings_current=confirmed_date == as_of,
            policy_id=policy["policy_id"],
            store_pointer_sha256=refs[pointer_path.as_posix()],
        )
        anchor_map = {row["symbol"]: row for row in policy["anchors"]}
        stop_map = {row["symbol"]: row for row in stops["stops"]}
        for symbol, position in sorted(current.items()):
            anchor = anchor_map.get(symbol)
            local = list(blockers[symbol])
            trailing_only = []
            try:
                if anchor and "anchor_ref" in anchor:
                    ref = anchor["anchor_ref"]
                    registered = {
                        str(RECORD_ROOT / row["manual_manifest_path"]): row[
                            "manual_manifest_sha256"
                        ]
                        for row in records.values()
                        if row.get("manual_manifest_path")
                    }
                    if registered.get(ref["path"]) != ref["sha256"]:
                        raise ResearchRiskError("ENTRY_REFERENCE_NOT_REGISTERED")
                    manual = json.loads(exact(ref["path"], ref["sha256"]))
                    trades = [
                        row
                        for key in (
                            "reconciled_source_trades",
                            "applied_owner_declared_trades",
                            "applied_local_trades",
                        )
                        for row in manual.get(key, [])
                        if row.get("symbol") == symbol
                    ]

                    def matches(row: dict[str, Any]) -> bool:
                        return (
                            decimal(row.get("shares")) == decimal(ref["shares"])
                            and decimal(row.get("execution_price"))
                            == decimal(ref["execution_price_cny"])
                            and decimal(row.get("final_total_fee_cny", row.get("fees_cny")))
                            == decimal(ref["final_total_fee_cny"])
                            and str(row.get("trade_date", manual.get("trade_date", ""))).replace(
                                "-", ""
                            )
                            == anchor["tracking_start_date"]
                        )

                    if not any(matches(row) for row in trades):
                        trailing_only.append("EXACT_ENTRY_REF_MISMATCH")
            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                trailing_only.append("ENTRY_REFERENCE_UNCONFIRMED:" + str(exc))
            stop = stop_map.get(symbol)
            start = anchor["tracking_start_date"] if anchor else target
            if stop:
                start = min(start, stops["effective_from"][:10].replace("-", ""))
            for partition in reader.table_partition_paths(start, target):
                relative_partition = partition.relative_to(project)
                market_files.add(relative_partition.as_posix())
                exact(relative_partition, refs.get(relative_partition.as_posix()))
            read = reader.read_symbol_frame(symbol, start_date=start, end_date=target)
            if read.issues or read.frame.empty:
                trailing_only.append("STRICT_CLOSE_UNAVAILABLE")
            stop_value = stop["initial_stop_price_cny"] if stop else None
            stop_blockers = []
            if stop:
                stop_start = stops["effective_from"][:10].replace("-", "")
                stop_rows = read.frame[read.frame["trade_date"] >= stop_start]
                if set(stop_rows["trade_date"]) != {d for d in expected_dates if d >= stop_start}:
                    stop_blockers.append("OWNER_STOP_HISTORY_GAP")
                if (
                    "adj_factor" not in stop_rows
                    or stop_rows["adj_factor"].nunique(dropna=False) != 1
                ):
                    stop_blockers.append("OWNER_STOP_CORPORATE_ACTION_REVIEW")
            if stop and (
                decimal(position["shares"]) != decimal(stop["current_shares"])
                or decimal(position["avg_cost"]) != decimal(stop["fee_inclusive_avg_cost_cny"])
            ):
                stop_blockers.append("OWNER_STOP_POSITION_CHANGED")
            row = calculate_position_risk(
                position=position,
                anchor=anchor,
                closes=read.frame.to_dict("records"),
                expected_dates=expected_dates,
                as_of=target,
                lifecycle_blockers=local,
                owner_stop=stop_value,
                holdings_current=confirmed_date == as_of,
                owner_stop_blockers=stop_blockers,
                trailing_blockers=trailing_only,
            )
            row["audit_ledger_thresholds"] = {
                key: position.get(key)
                for key in (
                    "stage_target_price",
                    "stage_stop_price",
                    "trailing_profit_review_price",
                    "trailing_profit_reduce_price",
                    "trailing_stop_price",
                    "trailing_take_profit_status",
                )
            }
            row["audit_ledger_thresholds"]["executable"] = False
            result["rows"].append(seal(row))
        result["status"] = (
            "READY"
            if confirmed_date == as_of and not any(row["blockers"] for row in result["rows"])
            else "PARTIAL"
        )
        if confirmed_date != as_of:
            result["blockers"].append("CURRENT_HOLDINGS_CONTINUITY_UNCONFIRMED")
        result["blockers"] = sorted(
            set(
                [
                    *result["blockers"],
                    *[
                        row["symbol"] + ":" + reason
                        for row in result["rows"]
                        for reason in row["blockers"]
                    ],
                ]
            )
        )
        for relative, sha in list(refs.items()):
            exact(relative, sha)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        result.update(status="BLOCKED", rows=[], blockers=[str(exc)])
    result["source_refs"] = [{"path": path, "sha256": sha} for path, sha in sorted(refs.items())]
    return seal(result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", default=str(PROJECT_ROOT))
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = assert_private_tmp(Path(args.output))
    result = build_risk_monitor(Path(args.workspace_root), as_of=args.as_of)
    raw = json.dumps(
        result, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    sha = immutable_write(output, raw, max_bytes=4 * 1024 * 1024)
    print(
        json.dumps(
            {
                "status": result["status"],
                "path": str(output),
                "sha256": sha,
                "blockers": result["blockers"],
            }
        )
    )
    return 2 if result["status"] == "BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
