"""Governed multi-day CN official-close transaction for Strategy Record Store-v3.

The transaction is fully offline.  Provider capture and immutable benchmark
publication happen earlier in the maintenance lane.  This module consumes only
exact pointer/generation/receipt inputs, prepares every missing open date, then
advances the Strategy Record Store pointer exactly once.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, time, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from quant_investor.market.cn_benchmark_store import (
    CNBenchmarkStoreError,
    load_generation as load_benchmarks,
)
from quant_investor.strategy_records.event_store import (
    StrategyEventStoreError,
    load_generation as load_events,
)
from quant_investor.strategy_records.performance import (
    MONEY_QUANTUM,
    UNIT_QUANTUM,
    build_manifest as build_performance_manifest,
    build_owner_declaration as build_performance_owner_declaration,
    build_performance_history_ref,
    decimal_text,
    extend_performance_rows,
    immutable_write,
    load_performance_history,
    validate_lineage_index,
    write_deterministic_parquet,
)
from quant_investor.strategy_records.store import (
    CATALOG_SCHEMA_V3,
    StrategyRecordConflict,
    StrategyRecordStoreError,
    canonical_json_bytes,
    content_sha256,
    load_registered_catalog,
    publish_catalog,
)

from close_cn_dashboard_official_valuation import (
    BATCH_PUBLICATION_CLASS,
    BATCH_PUBLICATION_REASON,
    build_record,
)
from cn_dashboard_common import validate_record

BATCH_PLAN_SCHEMA = "myquant.cn_official_close_batch_plan.v1"
BATCH_IMPLEMENTATION_VERSION = "2"
BATCH_RECEIPT_SCHEMA = "myquant.strategy_daily_close_receipt.v1"
BATCH_COMPLETION_SCHEMA = "myquant.cn_official_close_batch_completion.v1"
POLICY_SCHEMA = "myquant.cn_daily_official_close_policy.v1"
RETROSPECTIVE_SCHEMA = "myquant.cn_official_close_retrospective_owner_declaration.v1"
_SHA = re.compile(r"^[0-9a-f]{64}$")
_SHANGHAI = ZoneInfo("Asia/Shanghai")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read(path: Path, *, label: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise StrategyRecordStoreError(f"{label} is not a regular file")
    first = path.read_bytes()
    if first != path.read_bytes():
        raise StrategyRecordStoreError(f"{label} was unstable")
    return first


def _load_json(path: Path, *, expected_sha: str, label: str) -> dict[str, Any]:
    raw = _read(path, label=label)
    if _sha(raw) != expected_sha:
        raise StrategyRecordStoreError(f"{label} SHA mismatch")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StrategyRecordStoreError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise StrategyRecordStoreError(f"{label} is not an object")
    return value


def _policy(project: Path, relative: str, expected_sha: str) -> tuple[dict[str, Any], str]:
    path = project / relative
    value = _load_json(path, expected_sha=expected_sha, label="official-close policy")
    required_allowed = {
        "DAILY_NO_ACTION_CONTINUITY_RECEIPT",
        "OFFICIAL_VALUATION_RECORD",
        "PERFORMANCE_APPEND",
        "IMMUTABLE_CATALOG_GENERATION",
        "STRATEGY_RECORD_POINTER_CAS",
    }
    forbidden = value.get("forbidden")
    if (
        value.get("schema_id") != POLICY_SCHEMA
        or value.get("policy_id") != "cn-daily-official-close-policy-v1"
        or value.get("strategy_label") != "aggressive_tech_manufacturing"
        or value.get("record_root") != "results/strategy_records/CN/aggressive_tech_manufacturing"
        or value.get("revoked_at") is not None
        or not required_allowed.issubset(set(value.get("allowed_writes") or []))
        or not isinstance(forbidden, dict)
        or not all(
            forbidden.get(name) is True
            for name in (
                "broker_connection",
                "order_creation",
                "trade_execution",
                "unregistered_share_mutation",
                "unregistered_cash_mutation",
            )
        )
        or value.get("broker_order_trade_authority") is not False
        or value.get("actual_holdings_mutation_authority") is not False
    ):
        raise StrategyRecordStoreError("official-close policy scope/authority mismatch")
    return value, relative


def _retrospective(
    project: Path, relative: str | None, expected_sha: str | None
) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]]]:
    if relative is None and expected_sha is None:
        return None, {}
    if relative is None or expected_sha is None:
        raise StrategyRecordStoreError("retrospective declaration ref is incomplete")
    value = _load_json(
        project / relative,
        expected_sha=expected_sha,
        label="retrospective owner declaration",
    )
    if (
        value.get("schema_id") != RETROSPECTIVE_SCHEMA
        or value.get("owner") != "Maxwell"
        or value.get("retrospective_empty_event_closure_authorized") is not True
        or value.get("broker_order_trade_authority") is not False
    ):
        raise StrategyRecordStoreError("retrospective owner declaration is invalid")
    rows: dict[str, dict[str, Any]] = {}
    dimensions = (
        "executions",
        "orders",
        "fills",
        "funding",
        "cost_basis_changes",
        "corporate_actions",
        "manual_changes",
    )
    for row in value.get("dates") or []:
        if not isinstance(row, dict):
            raise StrategyRecordStoreError("retrospective date row is invalid")
        day = date.fromisoformat(str(row.get("trade_date"))).isoformat()
        if day in rows or any(row.get(name) != [] for name in dimensions):
            raise StrategyRecordStoreError("retrospective event dimensions are not closed empty")
        rows[day] = row
    return value, rows


def _pointer_sha(path: Path) -> str:
    return _sha(_read(path, label="Strategy Record pointer"))


def _inventory(directory: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total = 0
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
            raise StrategyRecordStoreError("batch record inventory contains an unsafe entry")
        raw = _read(path, label="batch record artifact")
        rows.append(
            {
                "path": path.name,
                "sha256": _sha(raw),
                "bytes": len(raw),
                "media_type": (
                    "application/vnd.apache.parquet"
                    if path.suffix == ".parquet"
                    else "text/csv" if path.suffix == ".csv" else "application/json"
                ),
            }
        )
        total += len(raw)
    inventory_raw = canonical_json_bytes(rows)
    return {
        "inventory": rows,
        "inventory_sha256": _sha(inventory_raw),
        "file_count": len(rows),
        "total_bytes": total,
    }


def _record_closure(record_root: Path, record_dir: Path) -> dict[str, Any]:
    manual_raw = _read(record_dir / "manual_execution_manifest.json", label="batch manual")
    manual = json.loads(manual_raw)
    return {
        "record_id": record_dir.name,
        "relative_path": record_dir.relative_to(record_root).as_posix(),
        "manifest_path": (record_dir / "manifest.json").relative_to(record_root).as_posix(),
        "manifest_sha256": _sha(_read(record_dir / "manifest.json", label="batch manifest")),
        "manual_manifest_path": (record_dir / "manual_execution_manifest.json")
        .relative_to(record_root)
        .as_posix(),
        "manual_manifest_sha256": _sha(manual_raw),
        "ledger_path": (record_dir / "ledger_after_manual_switch.parquet")
        .relative_to(record_root)
        .as_posix(),
        "ledger_sha256": _sha(
            _read(record_dir / "ledger_after_manual_switch.parquet", label="batch ledger")
        ),
        "pnl_path": (record_dir / "pnl_summary.csv").relative_to(record_root).as_posix(),
        "pnl_sha256": _sha(_read(record_dir / "pnl_summary.csv", label="batch pnl")),
        "financial_state_sha256": str(manual["financial_state_sha256"]),
    }


def _calendar_dates(path: Path, expected_sha: str) -> tuple[list[str], dict[str, Any]]:
    value = _load_json(path, expected_sha=expected_sha, label="Calendar receipt")
    rows = value.get("ordered_open_dates")
    if (
        value.get("schema_version") != "cn-close-session-receipt.v1"
        or value.get("status") != "TARGET_AUTHORIZED"
        or not isinstance(rows, list)
    ):
        raise StrategyRecordStoreError("Calendar receipt contract is invalid")
    dates = [date.fromisoformat(f"{row[:4]}-{row[4:6]}-{row[6:]}").isoformat() for row in rows]
    if dates != sorted(set(dates)):
        raise StrategyRecordStoreError("Calendar open dates are not canonical")
    return dates, value


def _market(project: Path, expected_sha: str) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    pointer_path = project / "data/parquet/cn/_latest.json"
    pointer = _load_json(pointer_path, expected_sha=expected_sha, label="Market pointer")
    if pointer.get("status") != "OK" or pointer.get("blockers") not in (None, []):
        raise StrategyRecordStoreError("Market pointer is not complete")
    manifest_path = Path(str(pointer.get("manifest_path")))
    if not manifest_path.is_absolute():
        manifest_path = project / manifest_path
    manifest_raw = _read(manifest_path, label="Market snapshot manifest")
    manifest = json.loads(manifest_raw)
    if manifest.get("snapshot_id") != pointer.get("snapshot_id") or manifest.get(
        "latest_complete_trade_date"
    ) != pointer.get("latest_complete_trade_date"):
        raise StrategyRecordStoreError("Market pointer/manifest closure mismatch")
    return pointer, manifest_path, manifest


def _market_evidence(
    *,
    project: Path,
    market_pointer: dict[str, Any],
    market_pointer_sha: str,
    market_manifest_path: Path,
    market_manifest: dict[str, Any],
    benchmark: dict[str, Any],
    compatibility_csv: Path,
    trade_date: str,
    symbols: list[str],
) -> dict[str, Any]:
    compact = trade_date.replace("-", "")
    serving_root = Path(str(market_manifest.get("derived_serving_root")))
    if not serving_root.is_absolute():
        serving_root = project / serving_root
    stocks: list[dict[str, Any]] = []
    for symbol in symbols:
        path = serving_root / f"symbol={symbol}" / "bars.parquet"
        raw = _read(path, label=f"Market serving {symbol}")
        frame = pd.read_parquet(path)
        if not {"trade_date", "close"}.issubset(frame.columns):
            raise StrategyRecordStoreError(f"Market serving columns missing: {symbol}")
        dates = frame["trade_date"].astype(str).str.replace("-", "", regex=False)
        exact = frame.loc[dates == compact]
        if len(exact) != 1 or not float(exact.iloc[0]["close"]) > 0:
            raise StrategyRecordStoreError(
                f"held-security exact close missing: {symbol}:{trade_date}"
            )
        stocks.append(
            {
                "symbol": symbol,
                "trade_date": compact,
                "close": float(exact.iloc[0]["close"]),
                "serving_parquet_path": path.relative_to(project).as_posix(),
                "serving_parquet_sha256": _sha(raw),
            }
        )
    benchmark_rows = [row for row in benchmark["rows"] if row["date"].isoformat() == trade_date]
    if len(benchmark_rows) != 3:
        raise StrategyRecordStoreError(f"benchmark exact close missing:{trade_date}")
    csv_raw = _read(compatibility_csv, label="benchmark compatibility projection")
    return {
        "schema_version": "cn_dashboard_strict_market_close_evidence.v1",
        "market": "CN",
        "trade_date": compact,
        "market_pointer_path": "data/parquet/cn/_latest.json",
        "market_pointer_sha256": market_pointer_sha,
        "snapshot_manifest_path": market_manifest_path.relative_to(project).as_posix(),
        "snapshot_manifest_sha256": _sha(_read(market_manifest_path, label="Market manifest")),
        "snapshot_id": market_pointer["snapshot_id"],
        "latest_complete_trade_date": market_pointer["latest_complete_trade_date"],
        "benchmark_input_path": compatibility_csv.relative_to(project).as_posix(),
        "benchmark_input_sha256": _sha(csv_raw),
        "benchmark_pointer_path": "data/parquet/cn/benchmarks/_latest.json",
        "benchmark_pointer_sha256": benchmark["pointer_sha256"],
        "benchmark_manifest_sha256": benchmark["manifest_sha256"],
        "benchmark_series_sha256": benchmark["series_sha256"],
        "stocks": stocks,
        "indices": [
            {
                "ts_code": row["ts_code"],
                "trade_date": compact,
                "close": float(row["close"]),
                "benchmark_input_path": compatibility_csv.relative_to(project).as_posix(),
                "benchmark_input_sha256": _sha(csv_raw),
            }
            for row in benchmark_rows
        ],
    }


def _write_exact_json(path: Path, value: Mapping[str, Any]) -> str:
    raw = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read(path, label="immutable batch artifact") != raw:
            raise StrategyRecordConflict("batch immutable identity collision")
        return _sha(raw)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        os.write(fd, raw)
        os.fsync(fd)
    finally:
        os.close(fd)
    return _sha(raw)


def _fingerprint(value: Mapping[str, Any]) -> str:
    return _sha(canonical_json_bytes(value))


def _completion_path(record_root: Path, transaction_id: str) -> Path:
    return (
        record_root
        / "_record_store/daily_close_transactions"
        / transaction_id
        / "completion.v1.json"
    )


def _write_completion(
    *, record_root: Path, plan: Mapping[str, Any], pointer_sha: str, status: str
) -> dict[str, Any]:
    observed = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    completion = {
        "schema_id": BATCH_COMPLETION_SCHEMA,
        "transaction_id": plan["transaction_id"],
        "input_fingerprint": plan["input_fingerprint"],
        "requested_target": plan["requested_target"],
        "committed_through": plan["requested_target"],
        "status": status,
        "effective_at": plan["effective_at"],
        "cas_observed_at": observed,
        "pointer_sha256": pointer_sha,
        "broker_order_trade_authority": False,
    }
    completion["content_sha256"] = content_sha256(completion)
    _write_exact_json(_completion_path(record_root, str(plan["transaction_id"])), completion)
    return completion


def close_through_latest(
    *,
    project_root: Path,
    record_root: Path,
    expected_store_pointer_sha: str,
    expected_market_pointer_sha: str,
    expected_benchmark_pointer_sha: str,
    expected_event_pointer_sha: str,
    calendar_receipt_path: Path,
    calendar_receipt_sha: str,
    policy_path: str,
    policy_sha: str,
    retrospective_path: str | None,
    retrospective_sha: str | None,
    execute: bool,
    now: datetime | None = None,
) -> dict[str, Any]:
    project = project_root.resolve(strict=True)
    root = record_root.resolve(strict=True)
    if _pointer_sha(root / "_record_store/current.v1.json") != expected_store_pointer_sha:
        raise StrategyRecordStoreError("Store pointer preimage mismatch")
    loaded = load_registered_catalog(root)
    if loaded is None:
        raise StrategyRecordStoreError("Store-v3 is unregistered")
    pointer, catalog = loaded
    if catalog.get("schema_id") != CATALOG_SCHEMA_V3:
        raise StrategyRecordStoreError("close-through-latest requires Store-v3")
    policy, _ = _policy(project, policy_path, policy_sha)
    _declaration, retrospective_rows = _retrospective(
        project, retrospective_path, retrospective_sha
    )
    market_pointer, market_manifest_path, market_manifest = _market(
        project, expected_market_pointer_sha
    )
    try:
        benchmark = load_benchmarks(project / "data/parquet/cn/benchmarks")
    except CNBenchmarkStoreError as exc:
        raise StrategyRecordStoreError(f"BENCHMARK_SOURCE_UNAVAILABLE:{exc}") from exc
    if benchmark["pointer_sha256"] != expected_benchmark_pointer_sha:
        raise StrategyRecordStoreError("benchmark pointer preimage mismatch")
    try:
        event = load_events(root / "_event_store")
    except StrategyEventStoreError as exc:
        raise StrategyRecordStoreError(f"EVENT_SOURCE_UNAVAILABLE:{exc}") from exc
    if event["pointer_sha256"] != expected_event_pointer_sha:
        raise StrategyRecordStoreError("event pointer preimage mismatch")
    open_dates, _calendar = _calendar_dates(calendar_receipt_path, calendar_receipt_sha)
    performance = load_performance_history(root, catalog["performance_history_ref"])
    official_date = str(performance["rows"][-1]["valuation_date"])
    market_end = date.fromisoformat(
        f"{str(market_pointer['latest_complete_trade_date'])[:4]}-{str(market_pointer['latest_complete_trade_date'])[4:6]}-{str(market_pointer['latest_complete_trade_date'])[6:]}"
    ).isoformat()
    required_candidates = [day for day in open_dates if official_date < day <= market_end]
    max_backlog = int(policy["max_backlog_open_days"])
    if len(required_candidates) > max_backlog:
        raise StrategyRecordStoreError("official-close backlog exceeds policy limit")
    if not required_candidates:
        active_id = str(pointer.get("active_record_id") or "")
        committed = [
            row
            for row in catalog.get("receipts", [])
            if isinstance(row, dict)
            and row.get("schema_id") == BATCH_RECEIPT_SCHEMA
            and row.get("record_id") == active_id
        ]
        recovered = None
        if len(committed) == 1:
            transaction_id = str(committed[0].get("transaction_id") or "")
            plan_path = (
                root / "_record_store/daily_close_transactions" / transaction_id / "plan.v1.json"
            )
            if plan_path.exists() and execute:
                plan = json.loads(_read(plan_path, label="daily-close frozen plan"))
                recovered = _write_completion(
                    record_root=root,
                    plan=plan,
                    pointer_sha=expected_store_pointer_sha,
                    status="RECOVERED_AFTER_CAS",
                )
        return {
            "status": "NO_ACTION",
            "last_official_date": official_date,
            "latest_required_close_date": market_end,
            "missing_dates": [],
            "pointer_sha256": expected_store_pointer_sha,
            "completion": recovered,
            "provider_calls": False,
            "broker_calls": False,
            "order_calls": False,
            "trade_calls": False,
        }
    closures = {row["trade_date"]: row for row in event["closures"]}
    for day in required_candidates:
        closure = closures.get(day)
        if closure is None:
            raise StrategyRecordStoreError(f"EVENT_STATE_CLOSURE_MISSING:{day}")
        if day in retrospective_rows:
            for name in (
                "executions",
                "orders",
                "fills",
                "funding",
                "cost_basis_changes",
                "corporate_actions",
                "manual_changes",
            ):
                if retrospective_rows[day].get(name) != []:
                    raise StrategyRecordStoreError(f"retrospective event is not empty:{day}")
    active_dir = root / str(pointer["active_record_id"])
    active_ledger = pd.read_parquet(active_dir / "ledger_after_manual_switch.parquet")
    symbols = sorted(active_ledger["symbol"].astype(str).tolist())
    if not symbols or len(symbols) != len(set(symbols)):
        raise StrategyRecordStoreError("active holdings symbol set is invalid")
    compatibility_csv = project / "portfolio_dashboard/inputs/cn_index_benchmark.csv"
    evidences = {
        day: _market_evidence(
            project=project,
            market_pointer=market_pointer,
            market_pointer_sha=expected_market_pointer_sha,
            market_manifest_path=market_manifest_path,
            market_manifest=market_manifest,
            benchmark=benchmark,
            compatibility_csv=compatibility_csv,
            trade_date=day,
            symbols=symbols,
        )
        for day in required_candidates
    }
    preimages = {
        "store_pointer_sha256": expected_store_pointer_sha,
        "store_catalog_sha256": pointer["catalog_sha256"],
        "performance_manifest_sha256": catalog["performance_history_ref"]["manifest"]["sha256"],
        "market_pointer_sha256": expected_market_pointer_sha,
        "benchmark_pointer_sha256": expected_benchmark_pointer_sha,
        "event_pointer_sha256": expected_event_pointer_sha,
        "calendar_receipt_sha256": calendar_receipt_sha,
        "policy_sha256": policy_sha,
        "retrospective_sha256": retrospective_sha,
        "evidence_sha256": {
            day: _sha(canonical_json_bytes(evidences[day])) for day in required_candidates
        },
    }
    input_fingerprint = _fingerprint(
        {
            "batch_implementation_version": BATCH_IMPLEMENTATION_VERSION,
            "requested_target": required_candidates[-1],
            "missing_dates": required_candidates,
            "preimages": preimages,
        }
    )
    transaction_id = (
        f"daily-close-{required_candidates[-1].replace('-', '')}-{input_fingerprint[:16]}"
    )
    transaction_root = root / "_record_store/daily_close_transactions" / transaction_id
    plan_path = transaction_root / "plan.v1.json"
    if plan_path.exists():
        plan_raw = _read(plan_path, label="daily-close frozen plan")
        plan = json.loads(plan_raw)
        if plan.get("input_fingerprint") != input_fingerprint:
            raise StrategyRecordConflict("daily-close frozen plan input conflict")
    else:
        planned = (
            (now or datetime.now(timezone.utc)).astimezone(timezone.utc).replace(microsecond=0)
        )
        shanghai = planned.astimezone(_SHANGHAI)
        record_ids = [
            f"{shanghai.strftime('%Y%m%d_%H%M%S')}-b{index:02d}"
            for index in range(1, len(required_candidates) + 1)
        ]
        effective_at = planned.isoformat().replace("+00:00", "Z")
        plan = {
            "schema_id": BATCH_PLAN_SCHEMA,
            "batch_implementation_version": BATCH_IMPLEMENTATION_VERSION,
            "transaction_id": transaction_id,
            "input_fingerprint": input_fingerprint,
            "transaction_planned_at": effective_at,
            "effective_at": effective_at,
            "source_active_record_id": pointer["active_record_id"],
            "last_official_date": official_date,
            "requested_target": required_candidates[-1],
            "missing_dates": required_candidates,
            "record_ids": record_ids,
            "catalog_generation_id": f"g-{transaction_id}",
            "performance_generation_id": f"p-{transaction_id}",
            "event_generation_id": event["pointer"]["generation_id"],
            "benchmark_generation_id": benchmark["pointer"]["generation_id"],
            "preimages": preimages,
            "publication_class": BATCH_PUBLICATION_CLASS,
            "all_or_nothing": True,
            "broker_order_trade_authority": False,
        }
        plan["content_sha256"] = content_sha256(plan)
        if execute:
            _write_exact_json(plan_path, plan)
    if not execute:
        return {
            "status": "PLAN_READY",
            "transaction_id": transaction_id,
            "last_official_date": official_date,
            "latest_required_close_date": required_candidates[-1],
            "missing_dates": required_candidates,
            "first_gap": None,
            "plan_path": plan_path.relative_to(project).as_posix(),
            "plan_sha256": _sha(canonical_json_bytes(plan)),
            "provider_calls": False,
            "broker_calls": False,
            "order_calls": False,
            "trade_calls": False,
        }
    existing_matches = [
        row
        for row in catalog.get("receipts", [])
        if isinstance(row, dict)
        and row.get("schema_id") == BATCH_RECEIPT_SCHEMA
        and row.get("transaction_id") == transaction_id
        and row.get("input_fingerprint") == input_fingerprint
    ]
    if existing_matches and performance["rows"][-1]["valuation_date"] == required_candidates[-1]:
        completion = _write_completion(
            record_root=root,
            plan=plan,
            pointer_sha=expected_store_pointer_sha,
            status="RECOVERED_AFTER_CAS",
        )
        return {
            "status": "NO_ACTION",
            "transaction_id": transaction_id,
            "missing_dates": [],
            "pointer_sha256": expected_store_pointer_sha,
            "completion": completion,
            "provider_calls": False,
            "broker_calls": False,
            "order_calls": False,
            "trade_calls": False,
        }
    staging_root = transaction_root / "records"
    staging_root.mkdir(parents=True, exist_ok=True)
    source_dir = active_dir
    source_closure = dict(pointer["active_closure"])
    source_record_dirs: dict[str, Path] = {
        str(pointer["active_record_id"]): active_dir,
    }
    built_dirs: list[Path] = []
    strict_rows: list[dict[str, Any]] = []
    daily_receipts: list[dict[str, Any]] = []
    for index, (day, record_id) in enumerate(zip(required_candidates, plan["record_ids"]), start=1):
        stage = staging_root / record_id
        stage.mkdir(exist_ok=True)
        closure = closures[day]
        closure_sha = str(closure["content_sha256"])
        event_receipt_id = f"daily-close/{day}/{closure_sha[:16]}"
        if not any(stage.iterdir()):
            build_record(
                staging_dir=stage,
                record_root=root,
                source_dir=source_dir,
                registered_closure=source_closure,
                record_id=record_id,
                trade_date=day.replace("-", ""),
                recorded_at_iso=plan["effective_at"],
                evidence=evidences[day],
                project_root=project,
                expected_market_pointer_sha256=expected_market_pointer_sha,
                source_pointer_sha256=expected_store_pointer_sha,
                source_catalog_generation_id=pointer["generation_id"],
                source_catalog_sha256=pointer["catalog_sha256"],
                continuity_receipt_id=event_receipt_id,
                continuity_receipt_sha256=closure_sha,
                continuity_receipt_created_at=closure["sealed_at"],
                continuity_checkpoint_digest=content_sha256(source_closure),
                evidence_input_sha256=_sha(canonical_json_bytes(evidences[day])),
                evidence_raw=canonical_json_bytes(evidences[day]),
                publication_class=BATCH_PUBLICATION_CLASS,
                expected_valuation_date=day,
                expected_publication_date=plan["effective_at"][:10],
                publication_delay_reason=BATCH_PUBLICATION_REASON,
            )
        strict = validate_record(
            stage,
            root,
            project,
            source_record_dirs=source_record_dirs,
        )
        if strict["data_date"] != day:
            raise StrategyRecordStoreError("batch record valuation date drifted")
        built_dirs.append(stage)
        strict_rows.append(strict)
        source_record_dirs[record_id] = stage
        source_dir = stage
        source_closure = _record_closure(root, stage)
        receipt = {
            "schema_id": BATCH_RECEIPT_SCHEMA,
            "receipt_id": event_receipt_id,
            "transaction_id": transaction_id,
            "input_fingerprint": input_fingerprint,
            "trade_date": day,
            "event_closure_sha256": closure_sha,
            "record_id": record_id,
            "status": "OFFICIAL_CLOSE_PREPARED",
            "effective_at": plan["effective_at"],
            "payload_copied": False,
            "actual_holdings_mutation_authority": False,
            "cash_mutation_authority": False,
            "broker_order_trade_authority": False,
        }
        receipt["content_sha256"] = content_sha256(receipt)
        daily_receipts.append(receipt)
    # Adopt complete immutable records.  A pre-CAS crash leaves recoverable orphans.
    adopted: list[Path] = []
    for stage, record_id in zip(built_dirs, plan["record_ids"]):
        target = root / record_id
        if target.exists():
            if _inventory(target) != _inventory(stage):
                raise StrategyRecordConflict("batch record adoption conflict")
        else:
            os.replace(stage, target)
        validate_record(target, root, project)
        adopted.append(target)
    records = [dict(row) for row in catalog["records"]]
    new_catalog_rows: list[dict[str, Any]] = []
    for target, strict in zip(adopted, strict_rows):
        closure = _record_closure(root, target)
        row = {
            "record_id": target.name,
            "relative_path": target.name,
            "state": "ONLINE",
            "storage_state": "ONLINE",
            "sealed_at": plan["effective_at"],
            **_inventory(target),
            **{
                key: value
                for key, value in closure.items()
                if key not in {"record_id", "relative_path"}
            },
            "history_eligible": True,
            "evidence_status": "HASH_VERIFIED",
            "summary": {
                "symbols": [position["symbol"] for position in strict["positions"]],
                "actions": [],
            },
        }
        if any(existing.get("record_id") == target.name for existing in records):
            old = next(existing for existing in records if existing.get("record_id") == target.name)
            if old != row:
                raise StrategyRecordConflict("batch catalog record collision")
        else:
            records.append(row)
        new_catalog_rows.append(row)
    next_performance = list(performance["rows"])
    lineage = [dict(row) for row in catalog["lineage_index"]]
    parent_id = str(pointer["active_record_id"])
    for strict, row in zip(strict_rows, new_catalog_rows):
        next_performance = extend_performance_rows(
            next_performance,
            strict_record=strict,
            manual_manifest_sha256=row["manual_manifest_sha256"],
            ledger_parquet_sha256=row["ledger_sha256"],
            financial_state_sha256=row["financial_state_sha256"],
            post_flow_unit_count=None,
            external_flow_amount=Decimal("0.0000"),
            allow_same_date_correction=False,
        )
        lineage.append(
            {
                "record_id": row["record_id"],
                "source_record_id": parent_id,
                "supersedes_record_id": None,
                "valuation_date": strict["data_date"],
                "execution_class": "NO_TRADE",
                "publication_class": BATCH_PUBLICATION_CLASS,
                "storage_state": "ONLINE",
                "manifest_ref": {"path": row["manifest_path"], "sha256": row["manifest_sha256"]},
                "manual_manifest_ref": {
                    "path": row["manual_manifest_path"],
                    "sha256": row["manual_manifest_sha256"],
                },
                "effective_ledger_ref": {
                    "path": row["ledger_path"],
                    "sha256": row["ledger_sha256"],
                },
                "financial_state_sha256": row["financial_state_sha256"],
                "ledger_parquet_sha256": row["ledger_sha256"],
            }
        )
        parent_id = row["record_id"]
    validate_lineage_index(lineage, active_record_id=new_catalog_rows[-1]["record_id"])
    performance_generation = str(plan["performance_generation_id"])
    prefix_text = f"_record_store/performance/{performance_generation}"
    prefix = root / prefix_text
    series_sha, series_bytes = (
        write_deterministic_parquet(next_performance, prefix / "series.parquet")
        if not (prefix / "series.parquet").exists()
        else (
            _sha(_read(prefix / "series.parquet", label="batch performance series")),
            len(_read(prefix / "series.parquet", label="batch performance series")),
        )
    )
    parent_manifest = performance["manifest"]
    owner = build_performance_owner_declaration(
        performance_generation_id=performance_generation,
        declared_at=plan["effective_at"],
        series_path=f"{prefix_text}/series.parquet",
        series_sha256=series_sha,
        series_bytes=series_bytes,
        source_pointer_sha256=expected_store_pointer_sha,
        source_catalog_sha256=pointer["catalog_sha256"],
        normalized_projection_semantic_sha256=parent_manifest[
            "normalized_projection_semantic_sha256"
        ],
    )
    owner_raw = canonical_json_bytes(owner)
    owner_sha = immutable_write(prefix / "owner_declaration.v1.json", owner_raw)
    performance_manifest = build_performance_manifest(
        performance_generation_id=performance_generation,
        generated_at=plan["effective_at"],
        identity_path=parent_manifest["identity_declaration"]["path"],
        identity_sha256=parent_manifest["identity_declaration"]["sha256"],
        parent_performance_manifest_sha256=catalog["performance_history_ref"]["manifest"]["sha256"],
        source_pointer_sha256=expected_store_pointer_sha,
        source_catalog_generation_id=pointer["generation_id"],
        source_catalog_sha256=pointer["catalog_sha256"],
        dashboard_projection_sha256=parent_manifest["source_dashboard_projection_sha256"],
        normalized_projection_semantic_sha256=parent_manifest[
            "normalized_projection_semantic_sha256"
        ],
        series_path=f"{prefix_text}/series.parquet",
        series_sha256=series_sha,
        series_bytes=series_bytes,
        owner_path=f"{prefix_text}/owner_declaration.v1.json",
        owner_sha256=owner_sha,
        owner_bytes=len(owner_raw),
        rows=next_performance,
    )
    manifest_raw = canonical_json_bytes(performance_manifest)
    manifest_sha = immutable_write(prefix / "manifest.v1.json", manifest_raw)
    performance_ref = build_performance_history_ref(
        manifest=performance_manifest,
        manifest_sha256=manifest_sha,
        manifest_bytes=len(manifest_raw),
    )
    load_performance_history(root, performance_ref)
    # Final preimage replay after every expensive candidate write/adoption.
    if (
        _pointer_sha(root / "_record_store/current.v1.json") != expected_store_pointer_sha
        or _sha(_read(project / "data/parquet/cn/_latest.json", label="Market pointer"))
        != expected_market_pointer_sha
        or load_benchmarks(project / "data/parquet/cn/benchmarks")["pointer_sha256"]
        != expected_benchmark_pointer_sha
        or load_events(root / "_event_store")["pointer_sha256"] != expected_event_pointer_sha
        or _sha(_read(calendar_receipt_path, label="Calendar receipt")) != calendar_receipt_sha
        or _sha(_read(project / policy_path, label="official-close policy")) != policy_sha
    ):
        raise StrategyRecordStoreError("daily-close preimage drift before Store CAS")
    published = publish_catalog(
        root,
        expected_pointer_sha256=expected_store_pointer_sha,
        records=records,
        receipts=daily_receipts,
        active_record_id=new_catalog_rows[-1]["record_id"],
        previous_record_id=(
            new_catalog_rows[-2]["record_id"]
            if len(new_catalog_rows) > 1
            else pointer["active_record_id"]
        ),
        generation_id=str(plan["catalog_generation_id"]),
        published_at=plan["effective_at"],
        catalog_schema=CATALOG_SCHEMA_V3,
        inherit_history_registry=False,
        lineage_index=lineage,
        performance_history_ref=performance_ref,
    )
    completion = _write_completion(
        record_root=root,
        plan=plan,
        pointer_sha=published["pointer_sha256"],
        status="COMMITTED",
    )
    return {
        "status": "COMMITTED",
        "transaction_id": transaction_id,
        "missing_dates": required_candidates,
        "committed_through": required_candidates[-1],
        "record_ids": plan["record_ids"],
        "catalog_generation_id": plan["catalog_generation_id"],
        "performance_generation_id": plan["performance_generation_id"],
        "pointer_sha256": published["pointer_sha256"],
        "completion": completion,
        "provider_calls": False,
        "broker_calls": False,
        "order_calls": False,
        "trade_calls": False,
    }


__all__ = ["close_through_latest"]
