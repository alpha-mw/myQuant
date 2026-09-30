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
import io
import json
import math
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
from quant_investor.strategy_records.close_coverage import analyze_close_coverage
from quant_investor.strategy_records.performance import (
    MAX_PERFORMANCE_JSON_BYTES,
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
    load_catalog_snapshot,
    load_catalog_snapshot_bytes,
    publish_catalog,
)

from close_cn_dashboard_official_valuation import (
    BATCH_PUBLICATION_CLASS,
    BATCH_PUBLICATION_REASON,
    build_record,
)
from cn_dashboard_common import validate_record
from quant_investor.strategy_records import close_plan_contracts as close_contracts

BATCH_PLAN_SCHEMA = "myquant.cn_official_close_batch_plan.v1"
BATCH_IMPLEMENTATION_VERSION = "3"
BATCH_RECEIPT_SCHEMA = "myquant.strategy_daily_close_receipt.v1"
BATCH_COMPLETION_SCHEMA = "myquant.cn_official_close_batch_completion.v1"
POLICY_SCHEMA = "myquant.cn_daily_official_close_policy.v1"
RETROSPECTIVE_SCHEMA = "myquant.cn_official_close_retrospective_owner_declaration.v1"
_SHA = re.compile(r"^[0-9a-f]{64}$")
_SHANGHAI = ZoneInfo("Asia/Shanghai")


class OfficialCloseInputsIncomplete(StrategyRecordStoreError):
    """All independent required-date gaps, without preparing or writing records."""

    def __init__(self, coverage: dict[str, Any]):
        self.coverage = coverage
        super().__init__(";".join(coverage["blockers"]))


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
    from quant_investor.strategy_records.daily_event_source import validate_official_close_policy

    validate_official_close_policy(value)
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
                "type": "file",
                "size": len(raw),
                "sha256": _sha(raw),
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
    from quant_investor.operations.strict_close_table_source import (
        STRICT_CLOSE_EVIDENCE_V2,
        StrictCloseTableSourceError,
        build_table_close_rows,
    )

    compact = trade_date.replace("-", "")
    manifest_raw = _read(market_manifest_path, label="Market manifest")
    try:
        stocks, table_partition_ref = build_table_close_rows(
            project,
            snapshot_id=str(market_pointer["snapshot_id"]),
            snapshot_manifest_path=market_manifest_path.relative_to(project).as_posix(),
            manifest_raw=manifest_raw,
            trade_date=compact,
            symbols=symbols,
        )
    except StrictCloseTableSourceError as exc:
        raise StrategyRecordStoreError(
            f"held-security exact close missing: {trade_date}: {exc}"
        ) from exc
    benchmark_rows = [row for row in benchmark["rows"] if row["date"].isoformat() == trade_date]
    if len(benchmark_rows) != 3:
        raise StrategyRecordStoreError(f"benchmark exact close missing:{trade_date}")
    csv_raw = _read(compatibility_csv, label="benchmark compatibility projection")
    return {
        "schema_version": STRICT_CLOSE_EVIDENCE_V2,
        "market": "CN",
        "trade_date": compact,
        "market_pointer_path": "data/parquet/cn/_latest.json",
        "market_pointer_sha256": market_pointer_sha,
        "snapshot_manifest_path": market_manifest_path.relative_to(project).as_posix(),
        "snapshot_manifest_sha256": _sha(manifest_raw),
        "snapshot_id": market_pointer["snapshot_id"],
        "latest_complete_trade_date": market_pointer["latest_complete_trade_date"],
        "benchmark_input_path": compatibility_csv.relative_to(project).as_posix(),
        "benchmark_input_sha256": _sha(csv_raw),
        "benchmark_pointer_path": "data/parquet/cn/benchmarks/_latest.json",
        "benchmark_pointer_sha256": benchmark["pointer_sha256"],
        "benchmark_generation_id": benchmark["pointer"]["generation_id"],
        "benchmark_manifest_path": benchmark["pointer"]["manifest"]["path"],
        "benchmark_manifest_sha256": benchmark["manifest_sha256"],
        "benchmark_series_path": benchmark["pointer"]["series"]["path"],
        "benchmark_series_sha256": benchmark["series_sha256"],
        "stocks": stocks,
        "table_partition_ref": table_partition_ref,
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
    from quant_investor.intelligence.storage import _atomic_no_replace

    raw = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read(path, label="immutable batch artifact") != raw:
            raise StrategyRecordConflict("batch immutable identity collision")
        return _sha(raw)
    parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    temporary = path.with_name("." + path.name + ".pending-" + secrets.token_hex(12))
    fd = None
    try:
        fd = os.open(
            temporary.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        offset = 0
        while offset < len(raw):
            written = os.write(fd, raw[offset:])
            if written <= 0:
                raise StrategyRecordStoreError("batch immutable short write")
            offset += written
        os.fsync(fd)
        os.close(fd)
        fd = None
        try:
            _atomic_no_replace(temporary, path, parent_fd=parent_fd)
        except FileExistsError:
            if _read(path, label="concurrent immutable batch artifact") != raw:
                raise StrategyRecordConflict("batch immutable identity collision") from None
        os.fsync(parent_fd)
    finally:
        if fd is not None:
            os.close(fd)
        try:
            os.unlink(temporary.name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        os.close(parent_fd)
    return _sha(raw)


def _fingerprint(value: Mapping[str, Any]) -> str:
    return _sha(canonical_json_bytes(value))


def _allocate_record_ids(catalog: Mapping[str, Any], planned: datetime, count: int) -> list[str]:
    """Reserve the next suffixes from registered records, not directories or time scans.

    Multiple catch-up transactions may finish in the same real second. The
    publication timestamp is unchanged; only the bNN suffix advances.
    """
    prefix = planned.astimezone(_SHANGHAI).strftime("%Y%m%d_%H%M%S") + "-b"
    used = [
        int(row["record_id"][-2:])
        for row in catalog["records"]
        if re.fullmatch(re.escape(prefix) + r"[0-9]{2}", row["record_id"])
    ]
    first = max(used, default=0) + 1
    if first + count - 1 > 99:
        raise StrategyRecordStoreError(
            "daily-close publication second has no available record slots"
        )
    return [f"{prefix}{index:02d}" for index in range(first, first + count)]


def _completion_path(record_root: Path, transaction_id: str, plan_version: int = 1) -> Path:
    close_contracts.version_number(plan_version)
    return (
        record_root
        / "_record_store/daily_close_transactions"
        / transaction_id
        / f"completion.v{plan_version}.json"
    )


def _plan_version(plan: Mapping[str, Any]) -> int:
    return close_contracts.validate_plan(dict(plan))


def _assert_version_files(root: Path, transaction_id: str, version: int) -> None:
    directory = _completion_path(root, transaction_id, version).parent
    other = 3 - version
    if any(
        (directory / name).exists() or (directory / name).is_symlink()
        for name in (f"plan.v{other}.json", f"completion.v{other}.json")
    ):
        raise StrategyRecordConflict("CLOSE_TRANSACTION_VERSION_CONFLICT")


def _registered_plan_fields(proof: dict, declaration_ref: dict) -> dict:
    declaration = proof["declaration"]
    return {
        "source_profile": close_contracts.PROFILE,
        "registered_event_declaration_ref": declaration_ref,
        "decision_baseline_pointer_ref": declaration["baseline_store_pointer_ref"],
        "decision_baseline_catalog_ref": declaration["baseline_catalog_ref"],
        "decision_baseline_record_id": declaration["baseline_record_id"],
        "source_valuation_date": declaration["trade_date"],
    }


def _registered_plan_proof(root: Path, plan: Mapping[str, Any], *, retained: bool = False) -> dict:
    from scripts.registered_daily_event_sources import read_declaration

    close_contracts.validate_plan(dict(plan))
    project = root.parents[3]
    proof = read_declaration(
        workspace=project, declaration_ref=plan["registered_event_declaration_ref"]
    )
    fields = _registered_plan_fields(proof, plan["registered_event_declaration_ref"])
    writer = proof["writer"]
    if (
        any(plan[k] != v for k, v in fields.items())
        or plan["preimages"]["store_pointer_sha256"]
        != proof["declaration"]["writer_store_pointer_ref"]["sha256"]
        or plan["preimages"]["store_catalog_sha256"] != writer["pointer"]["catalog_sha256"]
        or plan["preimages"]["performance_manifest_sha256"]
        != writer["catalog"]["performance_history_ref"]["manifest"]["sha256"]
        or plan["source_active_record_id"] != writer["record"]["record"]
        or plan["last_official_date"] != proof["baseline"]["record"]["data_date"]
        or plan["requested_target"] != proof["declaration"]["trade_date"]
    ):
        raise StrategyRecordConflict("REGISTERED_CLOSE_SOURCE_BINDING_MISMATCH")
    from quant_investor.strategy_records.registered_event_contracts import stamp

    if stamp(plan["effective_at"]) < stamp(proof["declaration"]["registered_at"]):
        raise StrategyRecordConflict("REGISTERED_CLOSE_PLAN_BEFORE_DECLARATION")
    if retained:
        directory = _completion_path(root, plan["transaction_id"], 2).parent
        for name, source in (
            ("source-pointer.v1.json", proof["declaration"]["writer_store_pointer_ref"]),
            ("decision-source-pointer.v1.json", proof["declaration"]["baseline_store_pointer_ref"]),
        ):
            raw = _read(directory / name, label="registered close source custody")
            if _sha(raw) != source["sha256"] or raw != _read(
                project / source["path"], label="original registered pointer"
            ):
                raise StrategyRecordConflict("REGISTERED_CLOSE_POINTER_CUSTODY_MISMATCH")
    return proof


def _retain_decision_pointer(root: Path, plan: Mapping[str, Any]) -> None:
    proof = _registered_plan_proof(root, plan)
    source = proof["declaration"]["baseline_store_pointer_ref"]
    raw = _read(root.parents[3] / source["path"], label="registered Decision baseline")
    path = _completion_path(root, plan["transaction_id"], 2).with_name(
        "decision-source-pointer.v1.json"
    )
    if _write_exact_json(path, json.loads(raw)) != source["sha256"]:
        raise StrategyRecordConflict("REGISTERED_CLOSE_DECISION_CUSTODY_MISMATCH")
    _registered_plan_proof(root, plan, retained=True)


def _registered_continuity(plan: Mapping[str, Any], proof: dict) -> dict:
    reference = plan["registered_event_declaration_ref"]
    return {
        "receipt_id": f"daily-close/{plan['requested_target']}/{reference['sha256'][:16]}",
        "receipt_sha256": reference["sha256"],
        "receipt_created_at": proof["declaration"]["registered_at"],
        "checkpoint_digest": content_sha256(proof["writer"]["pointer"]["active_closure"]),
    }


def _registered_receipt(plan: Mapping[str, Any], proof: dict) -> dict:
    continuity = _registered_continuity(plan, proof)
    receipt = {
        "schema_id": close_contracts.RECEIPT_V2,
        "receipt_id": continuity["receipt_id"],
        "transaction_id": plan["transaction_id"],
        "input_fingerprint": plan["input_fingerprint"],
        "trade_date": plan["requested_target"],
        "record_id": plan["record_ids"][0],
        "status": "OFFICIAL_CLOSE_PREPARED",
        "effective_at": plan["effective_at"],
        "payload_copied": False,
        "actual_holdings_mutation_authority": False,
        "cash_mutation_authority": False,
        "broker_order_trade_authority": False,
        "registered_event_declaration_ref": plan["registered_event_declaration_ref"],
        "source_profile": plan["source_profile"],
        "writer_active_checkpoint_digest": continuity["checkpoint_digest"],
        "decision_baseline_pointer_ref": plan["decision_baseline_pointer_ref"],
    }
    return {**receipt, "content_sha256": content_sha256(receipt)}


def _validate_registered_commit(
    root: Path, catalog: Mapping[str, Any], plan: Mapping[str, Any], history: dict
) -> None:
    proof = _registered_plan_proof(root, plan, retained=True)
    receipts = [r for r in catalog["receipts"] if r.get("transaction_id") == plan["transaction_id"]]
    if receipts != [_registered_receipt(plan, proof)]:
        raise StrategyRecordConflict("REGISTERED_CLOSE_RECEIPT_MISMATCH")
    record_id = plan["record_ids"][0]
    record = next(r for r in catalog["records"] if r["record_id"] == record_id)
    final = validate_record(root / record_id, root, root.parents[3])
    expected = extend_performance_rows(
        proof["writer"]["history"]["rows"],
        strict_record=final,
        manual_manifest_sha256=record["manual_manifest_sha256"],
        ledger_parquet_sha256=record["ledger_sha256"],
        financial_state_sha256=record["financial_state_sha256"],
        official_close_source=proof["writer"]["record"],
    )
    if expected != history["rows"]:
        raise StrategyRecordConflict("REGISTERED_CLOSE_PERFORMANCE_REPLAY_MISMATCH")
    lineage = catalog["lineage_index"]
    if lineage[:-1] != proof["writer"]["catalog"]["lineage_index"] or (
        lineage[-1]["record_id"] != record_id
        or lineage[-1]["source_record_id"] != plan["source_active_record_id"]
        or lineage[-1]["supersedes_record_id"] is not None
        or lineage[-1]["execution_class"] != "NO_TRADE"
        or lineage[-1]["publication_class"] != BATCH_PUBLICATION_CLASS
    ):
        raise StrategyRecordConflict("REGISTERED_CLOSE_LINEAGE_MISMATCH")
    values = _registered_continuity(plan, proof)
    fields = {
        "continuity_receipt_id": values["receipt_id"],
        "continuity_receipt_sha256": values["receipt_sha256"],
        "continuity_receipt_created_at": values["receipt_created_at"],
        "continuity_checkpoint_digest": values["checkpoint_digest"],
        "source_record": plan["source_active_record_id"],
    }
    for name in ("manifest", "manual_manifest"):
        document = _load_json(
            root / record[name + "_path"],
            expected_sha=record[name + "_sha256"],
            label="registered final record",
        )
        if any(document.get(k) != v for k, v in fields.items()):
            raise StrategyRecordConflict("REGISTERED_CLOSE_CONTINUITY_MISMATCH")


def _adopt_registered_close(
    *,
    project: Path,
    root: Path,
    arguments: dict,
    declaration_proof: dict,
    declaration_ref: dict,
    current_sha: str,
) -> dict:
    from quant_investor.operations.daily_contract import validate_ref

    writer_sha = declaration_proof["declaration"]["writer_store_pointer_ref"]["sha256"]
    if arguments["expected_store_pointer_sha"] not in {writer_sha, current_sha}:
        raise StrategyRecordConflict("REGISTERED_CLOSE_ADOPTION_PREIMAGE_MISMATCH")
    pointer, catalog = load_registered_catalog(root)
    rows = [
        r
        for r in catalog["receipts"]
        if r.get("record_id") == pointer["active_record_id"]
        and r.get("schema_id") == close_contracts.RECEIPT_V2
    ]
    if len(rows) != 1:
        raise StrategyRecordConflict("REGISTERED_CLOSE_ADOPTION_RECEIPT_UNPROVEN")
    transaction = rows[0]["transaction_id"]
    if re.fullmatch(r"daily-close-[0-9]{8}-[0-9a-f]{16}", transaction) is None:
        raise StrategyRecordConflict("REGISTERED_CLOSE_ADOPTION_TRANSACTION_INVALID")
    path = _completion_path(root, transaction, 2).with_name("plan.v2.json")
    raw = _read(path, label="registered adoption plan")
    plan = _load_json(path, expected_sha=_sha(raw), label="registered adoption plan")
    proof = inspect_frozen_close_commit(
        record_root=root,
        transaction_id=transaction,
        expected_plan_sha=_sha(raw),
        expected_source_pointer_sha=writer_sha,
        expected_target=declaration_proof["declaration"]["trade_date"],
        plan_version=2,
    )
    names = {
        "expected_market_pointer_sha": "market_pointer_sha256",
        "expected_benchmark_pointer_sha": "benchmark_pointer_sha256",
        "expected_event_pointer_sha": "event_pointer_sha256",
        "calendar_receipt_sha": "calendar_receipt_sha256",
        "policy_sha": "policy_sha256",
        "retrospective_sha": "retrospective_sha256",
    }
    if (
        proof["pointer_sha256"] != current_sha
        or plan["registered_event_declaration_ref"] != declaration_ref
        or any(arguments[k] != plan["preimages"][v] for k, v in names.items())
    ):
        raise StrategyRecordConflict("REGISTERED_CLOSE_ADOPTION_BINDING_MISMATCH")
    if arguments.get("expected_plan_sha") is not None and arguments["expected_plan_sha"] != _sha(
        raw
    ):
        raise StrategyRecordConflict("daily-close prepared plan SHA differs")
    market, _, _ = _market(project, arguments["expected_market_pointer_sha"])
    _days, calendar = _calendar_dates(
        arguments["calendar_receipt_path"], arguments["calendar_receipt_sha"]
    )
    if str(market["latest_complete_trade_date"]) != plan["requested_target"].replace(
        "-", ""
    ) or str(calendar.get("target_trade_date", "")).replace("-", "") != plan[
        "requested_target"
    ].replace(
        "-", ""
    ):
        raise StrategyRecordConflict("REGISTERED_CLOSE_HISTORICAL_FINALIZATION_UNSUPPORTED")
    if (
        load_benchmarks(project / "data/parquet/cn/benchmarks")["pointer_sha256"]
        != arguments["expected_benchmark_pointer_sha"]
        or load_events(root / "_event_store")["pointer_sha256"]
        != arguments["expected_event_pointer_sha"]
        or _pointer_sha(root / "_record_store/current.v1.json") != current_sha
    ):
        raise StrategyRecordConflict("REGISTERED_CLOSE_ADOPTION_CURRENT_CHANGED")
    _policy(project, arguments["policy_path"], arguments["policy_sha"])
    source = proof["decision_source_pointer_ref"]
    return {
        "status": "PLAN_ADOPTED",
        "transaction_id": transaction,
        "plan_path": path.relative_to(project).as_posix(),
        "plan_sha256": _sha(raw),
        "native_plan": plan,
        "commit_proof": proof,
        "source_pointer_ref": validate_ref(
            {"path": str((root / source["path"]).relative_to(project)), "sha256": source["sha256"]}
        ),
        "latest_required_close_date": plan["requested_target"],
        "write_performed": False,
    }


def _write_completion(
    *, record_root: Path, plan: Mapping[str, Any], pointer_sha: str, status: str
) -> dict[str, Any]:
    # Retain the exact committed bytes before the terminal receipt. Future-day
    # replay must not depend on a mutable current pointer still being this day.
    raw = _read(record_root / "_record_store/current.v1.json", label="committed Store pointer")
    if _sha(raw) != pointer_sha:
        raise StrategyRecordConflict("Store pointer moved before completion custody")
    version = _plan_version(plan)
    retained = _completion_path(record_root, str(plan["transaction_id"]), version).with_name(
        "committed-pointer.v1.json"
    )
    if _write_exact_json(retained, json.loads(raw)) != pointer_sha:
        raise StrategyRecordConflict("retained committed pointer bytes differ")
    completion = _completion_value(plan, pointer_sha, status)
    _validate_close_completion(completion, plan)
    _write_exact_json(
        _completion_path(record_root, str(plan["transaction_id"]), version), completion
    )
    return completion


def _completion_value(plan: Mapping[str, Any], pointer_sha: str, status: str) -> dict[str, Any]:
    observed = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    completion = {
        "schema_id": (
            BATCH_COMPLETION_SCHEMA if _plan_version(plan) == 1 else close_contracts.COMPLETION_V2
        ),
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
    if _plan_version(plan) == 2:
        completion.update(
            registered_event_declaration_ref=plan["registered_event_declaration_ref"],
            decision_baseline_pointer_ref=plan["decision_baseline_pointer_ref"],
        )
    completion["content_sha256"] = content_sha256(completion)
    return completion


def _inspect_close_commit(
    *,
    record_root: Path,
    transaction_id: str,
    expected_plan_sha: str,
    expected_source_pointer_sha: str,
    expected_target: str,
    _frozen: bool = False,
    plan_version: int = 1,
) -> dict[str, Any]:
    """Read-only native commit proof, including after the current head advances.

    A prepared plan is not proof of CAS. Only its exact committed catalog and
    financial closure can establish that publication occurred.
    """
    if re.fullmatch(r"daily-close-[0-9]{8}-[0-9a-f]{16}", transaction_id) is None:
        raise StrategyRecordStoreError("daily-close transaction identity invalid")
    _assert_version_files(record_root, transaction_id, plan_version)
    completion_path = _completion_path(record_root, transaction_id, plan_version)
    plan_path = completion_path.with_name(f"plan.v{plan_version}.json")
    frozen_files = {}
    if _frozen:
        from quant_investor.system.storage import SecureSystemStorage

        reader = SecureSystemStorage(record_root)
        custody_paths = [
            plan_path,
            completion_path,
            completion_path.with_name("committed-pointer.v1.json"),
            completion_path.with_name("source-pointer.v1.json"),
        ]
        if plan_version == 2:
            custody_paths.append(completion_path.with_name("decision-source-pointer.v1.json"))
        for path in custody_paths:
            relative = path.relative_to(record_root).as_posix()
            frozen_files[relative] = reader.read_workspace_file_bytes(
                relative, maximum_bytes=16 * 1024 * 1024
            )
    plan = _load_json(plan_path, expected_sha=expected_plan_sha, label="committed close plan")
    if close_contracts.validate_plan(plan, path=plan_path) != plan_version:
        raise StrategyRecordConflict("CLOSE_PLAN_PATH_VERSION_MISMATCH")
    if (
        plan.get("schema_id")
        != (BATCH_PLAN_SCHEMA if plan_version == 1 else close_contracts.PLAN_V2)
        or plan.get("transaction_id") != transaction_id
        or plan.get("content_sha256") != content_sha256(plan)
        or plan.get("requested_target") != expected_target
        or plan.get("preimages", {}).get("store_pointer_sha256") != expected_source_pointer_sha
        or plan.get("broker_order_trade_authority") is not False
        or plan.get("all_or_nothing") is not True
    ):
        raise StrategyRecordConflict("daily-close committed plan binding differs")
    completion = None
    if completion_path.exists():
        completion = json.loads(_read(completion_path, label="close completion"))
        _validate_close_completion(completion, plan)
    retained = completion_path.with_name("committed-pointer.v1.json")
    if _frozen and (completion is None or not retained.exists()):
        raise StrategyRecordStoreError("frozen close commit custody incomplete")
    source = _read_close_source_pointer(record_root, plan) if _frozen else None
    pointer_path = retained if retained.exists() else record_root / "_record_store/current.v1.json"
    pointer_raw = _read(pointer_path, label="committed pointer evidence")
    pointer_sha = _sha(pointer_raw)
    if completion is not None and completion["pointer_sha256"] != pointer_sha:
        raise StrategyRecordStoreError("committed pointer evidence unavailable")
    pointer, catalog = load_catalog_snapshot(
        record_root,
        pointer_relative_path=pointer_path.relative_to(record_root).as_posix(),
        expected_pointer_sha256=pointer_sha,
    )
    if (
        catalog.get("schema_id") != CATALOG_SCHEMA_V3
        or pointer["generation_id"] != plan["catalog_generation_id"]
        or pointer["active_record_id"] != plan["record_ids"][-1]
        or pointer["previous_pointer_sha256"] != expected_source_pointer_sha
    ):
        raise StrategyRecordConflict("daily-close CAS not observed for prepared transaction")
    _validate_close_receipts_and_holdings(record_root, catalog, plan)
    if _frozen:
        assert source is not None
        from quant_investor.strategy_records.store import _active_closure

        if (
            _active_closure(catalog["records"], plan["source_active_record_id"])
            != source["pointer"]["active_closure"]
        ):
            raise StrategyRecordConflict("frozen close source checkpoint differs")
    else:
        _assert_registered_ancestor(record_root, pointer["active_record_id"])
    performance = load_performance_history(record_root, catalog["performance_history_ref"])
    if str(performance["rows"][-1]["valuation_date"]) != expected_target:
        raise StrategyRecordConflict("daily-close committed performance date differs")
    if plan_version == 2:
        _validate_registered_commit(record_root, catalog, plan, performance)
    if _frozen:
        for relative, original in frozen_files.items():
            again = reader.read_workspace_file_bytes(relative, maximum_bytes=16 * 1024 * 1024)
            if (again.data, again.stat_identity) != (original.data, original.stat_identity):
                raise StrategyRecordConflict("frozen close custody files changed")
        if (
            _load_json(plan_path, expected_sha=expected_plan_sha, label="frozen plan recheck")
            != plan
        ):
            raise StrategyRecordConflict("frozen close plan changed")
        if _read(pointer_path, label="frozen committed pointer recheck") != pointer_raw:
            raise StrategyRecordConflict("frozen committed pointer changed")
        if json.loads(_read(completion_path, label="frozen completion recheck")) != completion:
            raise StrategyRecordConflict("frozen completion changed")
        if _read_close_source_pointer(record_root, plan) != source:
            raise StrategyRecordConflict("frozen source pointer changed")
    return {
        "status": "VERIFIED" if completion else "COMMITTED_COMPLETION_MISSING",
        "plan": plan,
        "completion": completion,
        "pointer_sha256": pointer_sha,
        "pointer_ref": {
            "path": pointer_path.relative_to(record_root).as_posix(),
            "sha256": pointer_sha,
        },
        "catalog_ref": {"path": pointer["catalog_path"], "sha256": pointer["catalog_sha256"]},
        "performance_ref": catalog["performance_history_ref"],
        "write_performed": False,
        **({"source_pointer_ref": source["ref"]} if source is not None else {}),
        **(
            {
                "decision_source_pointer_ref": {
                    "path": completion_path.with_name("decision-source-pointer.v1.json")
                    .relative_to(record_root)
                    .as_posix(),
                    "sha256": plan["decision_baseline_pointer_ref"]["sha256"],
                }
            }
            if plan_version == 2
            else {}
        ),
    }


def inspect_close_commit(
    *,
    record_root: Path,
    transaction_id: str,
    expected_plan_sha: str,
    expected_source_pointer_sha: str,
    expected_target: str,
    plan_version: int = 1,
) -> dict[str, Any]:
    return _inspect_close_commit(
        record_root=record_root,
        transaction_id=transaction_id,
        expected_plan_sha=expected_plan_sha,
        expected_source_pointer_sha=expected_source_pointer_sha,
        expected_target=expected_target,
        plan_version=plan_version,
    )


def inspect_frozen_close_commit(
    *,
    record_root: Path,
    transaction_id: str,
    expected_plan_sha: str,
    expected_source_pointer_sha: str,
    expected_target: str,
    plan_version: int = 1,
) -> dict[str, Any]:
    """Full immutable commit/source proof; no current registration lookup or writer."""
    return _inspect_close_commit(
        record_root=record_root,
        transaction_id=transaction_id,
        expected_plan_sha=expected_plan_sha,
        expected_source_pointer_sha=expected_source_pointer_sha,
        expected_target=expected_target,
        _frozen=True,
        plan_version=plan_version,
    )


def _read_close_source_pointer(root: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    from quant_investor.system.storage import SecureSystemStorage

    path = _completion_path(root, str(plan["transaction_id"])).with_name("source-pointer.v1.json")
    raw = (
        SecureSystemStorage(root)
        .read_workspace_file_bytes(
            path.relative_to(root).as_posix(), maximum_bytes=16 * 1024 * 1024
        )
        .data
    )
    digest = plan["preimages"]["store_pointer_sha256"]
    pointer, catalog = load_catalog_snapshot_bytes(
        root, pointer_bytes=raw, expected_pointer_sha256=digest
    )
    if (
        pointer["catalog_sha256"] != plan["preimages"]["store_catalog_sha256"]
        or pointer["active_record_id"] != plan["source_active_record_id"]
    ):
        raise StrategyRecordConflict("native close source pointer plan mismatch")
    return {
        "pointer": pointer,
        "catalog": catalog,
        "ref": {"path": path.relative_to(root).as_posix(), "sha256": digest},
    }


def _retain_close_source_pointer(root: Path, plan: Mapping[str, Any]) -> None:
    raw = _read(root / "_record_store/current.v1.json", label="native close source preimage")
    expected = plan["preimages"]["store_pointer_sha256"]
    if _sha(raw) != expected:
        raise StrategyRecordConflict("native close source preimage changed before retention")
    pointer, _ = load_catalog_snapshot_bytes(
        root, pointer_bytes=raw, expected_pointer_sha256=expected
    )
    if (
        pointer["catalog_sha256"] != plan["preimages"]["store_catalog_sha256"]
        or pointer["active_record_id"] != plan["source_active_record_id"]
    ):
        raise StrategyRecordConflict("native close source preimage plan mismatch")
    path = _completion_path(root, str(plan["transaction_id"])).with_name("source-pointer.v1.json")
    if _write_exact_json(path, json.loads(raw)) != expected:
        raise StrategyRecordConflict("native close retained source bytes differ")
    _read_close_source_pointer(root, plan)
    if _read(root / "_record_store/current.v1.json", label="retained source recheck") != raw:
        raise StrategyRecordConflict("native close source changed during retention")


def _assert_registered_ancestor(root: Path, record_id: str) -> None:
    current = load_registered_catalog(root)
    if current is None:
        raise StrategyRecordStoreError("current Store registration unavailable")
    pointer, catalog = current
    rows = {row["record_id"]: row for row in catalog["lineage_index"]}
    cursor = pointer["active_record_id"]
    seen: set[str] = set()
    while cursor is not None and cursor not in seen:
        if cursor == record_id:
            return
        seen.add(cursor)
        cursor = rows[cursor]["source_record_id"]
    raise StrategyRecordConflict("prepared catalog is not in registered active ancestry")


def recover_close_completion(
    *,
    record_root: Path,
    transaction_id: str,
    expected_plan_sha: str,
    expected_source_pointer_sha: str,
    expected_target: str,
    plan_version: int = 1,
    _metadata_guard=None,
) -> dict[str, Any]:
    """Metadata-only recovery after native CAS proof; never publish a catalog."""
    proof = inspect_close_commit(
        record_root=record_root,
        transaction_id=transaction_id,
        expected_plan_sha=expected_plan_sha,
        expected_source_pointer_sha=expected_source_pointer_sha,
        expected_target=expected_target,
        plan_version=plan_version,
    )
    path = _completion_path(record_root, transaction_id, plan_version)
    retained = path.with_name("committed-pointer.v1.json")
    wrote = False
    value = proof["completion"]
    if value is None:
        value = _completion_value(proof["plan"], proof["pointer_sha256"], "RECOVERED_AFTER_CAS")
        _validate_close_completion(value, proof["plan"])
    pointer_raw = None
    if not retained.exists():
        pointer_raw = _read(
            record_root / proof["pointer_ref"]["path"], label="verified committed pointer"
        )
        if _sha(pointer_raw) != proof["pointer_sha256"]:
            raise StrategyRecordConflict("committed pointer changed during metadata recovery")
    # The ordinary financial API keeps its existing scope. A registered DAG
    # caller supplies this private code-owned guard while holding its original
    # package, day, strategy and Store locks. It runs before either metadata write
    # and cannot select or backdate the native observation timestamp.
    if _metadata_guard is not None:
        _metadata_guard(proof=proof, completion=value)
    if pointer_raw is not None:
        if _write_exact_json(retained, json.loads(pointer_raw)) != proof["pointer_sha256"]:
            raise StrategyRecordConflict("retained recovery pointer bytes differ")
        wrote = True
    if proof["completion"] is None:
        _write_exact_json(path, value)
        wrote = True
    final = inspect_close_commit(
        record_root=record_root,
        transaction_id=transaction_id,
        expected_plan_sha=expected_plan_sha,
        expected_source_pointer_sha=expected_source_pointer_sha,
        expected_target=expected_target,
        plan_version=plan_version,
    )
    return {**final, "metadata_write_performed": wrote, "catalog_cas_performed": False}


def _validate_close_completion(completion: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    if _plan_version(plan) == 2 and set(completion) != {
        "schema_id",
        "transaction_id",
        "input_fingerprint",
        "requested_target",
        "committed_through",
        "status",
        "effective_at",
        "cas_observed_at",
        "pointer_sha256",
        "broker_order_trade_authority",
        "content_sha256",
        "registered_event_declaration_ref",
        "decision_baseline_pointer_ref",
    }:
        raise StrategyRecordConflict("REGISTERED_CLOSE_COMPLETION_FIELDS_INVALID")
    if (
        completion.get("schema_id")
        != (BATCH_COMPLETION_SCHEMA if _plan_version(plan) == 1 else close_contracts.COMPLETION_V2)
        or completion.get("status") not in {"COMMITTED", "RECOVERED_AFTER_CAS"}
        or completion.get("content_sha256") != content_sha256(completion)
        or completion.get("broker_order_trade_authority") is not False
    ):
        raise StrategyRecordConflict("daily-close completion is invalid")
    for field in ("transaction_id", "input_fingerprint", "requested_target", "effective_at"):
        if completion.get(field) != plan.get(field):
            raise StrategyRecordConflict("daily-close completion plan binding differs")
    if completion.get("committed_through") != plan["requested_target"]:
        raise StrategyRecordConflict("daily-close completion target differs")
    if _plan_version(plan) == 2 and any(
        completion.get(k) != plan[k]
        for k in ("registered_event_declaration_ref", "decision_baseline_pointer_ref")
    ):
        raise StrategyRecordConflict("REGISTERED_CLOSE_COMPLETION_BINDING_MISMATCH")
    try:
        observed = datetime.strptime(completion["cas_observed_at"], "%Y-%m-%dT%H:%M:%SZ")
        planned = datetime.strptime(plan["effective_at"], "%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, KeyError, TypeError) as exc:
        raise StrategyRecordConflict("daily-close completion clock invalid") from exc
    if observed < planned:
        raise StrategyRecordConflict("daily-close completion precedes plan")


def _holdings_identity(root: Path, record: Mapping[str, Any]) -> tuple[pd.DataFrame, Decimal]:
    from quant_investor.strategy_records.holdings import load_holdings_identity

    return load_holdings_identity(root, record)


def _validate_close_receipts_and_holdings(
    root: Path, catalog: Mapping[str, Any], plan: Mapping[str, Any]
) -> None:
    receipts = [
        row for row in catalog["receipts"] if row.get("transaction_id") == plan["transaction_id"]
    ]
    expected = set(zip(plan["missing_dates"], plan["record_ids"]))
    observed = {(row.get("trade_date"), row.get("record_id")) for row in receipts}
    if not expected or len(receipts) != len(expected) or observed != expected:
        raise StrategyRecordConflict("daily-close committed date receipt set differs")
    for row in receipts:
        if (
            row.get("schema_id")
            != (BATCH_RECEIPT_SCHEMA if _plan_version(plan) == 1 else close_contracts.RECEIPT_V2)
            or row.get("input_fingerprint") != plan["input_fingerprint"]
            or row.get("content_sha256") != content_sha256(row)
            or any(
                row.get(flag) is not False
                for flag in (
                    "actual_holdings_mutation_authority",
                    "cash_mutation_authority",
                    "broker_order_trade_authority",
                )
            )
        ):
            raise StrategyRecordConflict("daily-close receipt authority or fingerprint differs")
    records = {row["record_id"]: row for row in catalog["records"]}
    before, cash = _holdings_identity(root, records[plan["source_active_record_id"]])
    for record_id in plan["record_ids"]:
        after, final_cash = _holdings_identity(root, records[record_id])
        if not before.equals(after) or final_cash != cash:
            raise StrategyRecordConflict("daily-close changed actual holdings or cash")


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
    prepare_only: bool = False,
    expected_plan_sha: str | None = None,
    registered_event_declaration_ref: dict[str, str] | None = None,
) -> dict[str, Any]:
    if type(execute) is not bool or type(prepare_only) is not bool or (prepare_only and execute):
        raise StrategyRecordStoreError("prepare-only and execute are distinct phases")
    project = project_root.resolve(strict=True)
    root = record_root.resolve(strict=True)
    registered = None
    version = 1
    if registered_event_declaration_ref is not None:
        if retrospective_path is not None or retrospective_sha is not None:
            raise StrategyRecordStoreError("REGISTERED_CLOSE_RETROSPECTIVE_PROFILE_CONFLICT")
        from quant_investor.operations.daily_contract import validate_ref
        from scripts.registered_daily_event_sources import read_declaration

        if root != project / "results/strategy_records/CN/aggressive_tech_manufacturing":
            raise StrategyRecordStoreError("REGISTERED_CLOSE_ROOT_INVALID")
        registered_event_declaration_ref = validate_ref(registered_event_declaration_ref)
        registered = read_declaration(
            workspace=project, declaration_ref=registered_event_declaration_ref
        )
        version = 2
        current_sha = _pointer_sha(root / "_record_store/current.v1.json")
        writer_sha = registered["declaration"]["writer_store_pointer_ref"]["sha256"]
        if current_sha != writer_sha:
            return _adopt_registered_close(
                project=project,
                root=root,
                arguments={
                    "expected_store_pointer_sha": expected_store_pointer_sha,
                    "expected_market_pointer_sha": expected_market_pointer_sha,
                    "expected_benchmark_pointer_sha": expected_benchmark_pointer_sha,
                    "expected_event_pointer_sha": expected_event_pointer_sha,
                    "calendar_receipt_path": calendar_receipt_path,
                    "calendar_receipt_sha": calendar_receipt_sha,
                    "policy_path": policy_path,
                    "policy_sha": policy_sha,
                    "retrospective_sha": retrospective_sha,
                    "expected_plan_sha": expected_plan_sha,
                },
                declaration_proof=registered,
                declaration_ref=registered_event_declaration_ref,
                current_sha=current_sha,
            )
    if _pointer_sha(root / "_record_store/current.v1.json") != expected_store_pointer_sha:
        raise StrategyRecordStoreError("Store pointer preimage mismatch")
    loaded = load_registered_catalog(root)
    if loaded is None:
        raise StrategyRecordStoreError("Store-v3 is unregistered")
    pointer, catalog = loaded
    if catalog.get("schema_id") != CATALOG_SCHEMA_V3:
        raise StrategyRecordStoreError("close-through-latest requires Store-v3")
    active_manual = _load_json(
        root / pointer["active_closure"]["manual_manifest_path"],
        expected_sha=pointer["active_closure"]["manual_manifest_sha256"],
        label="active official-close state",
    )
    if registered is None and active_manual.get("official_valuation") is False:
        raise StrategyRecordStoreError("REGISTERED_CLOSE_DECLARATION_REQUIRED")
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
    if not open_dates or market_end > open_dates[-1]:
        raise StrategyRecordStoreError("OFFICIAL_CLOSE_CALENDAR_BEHIND_MARKET:" + market_end)
    required_candidates = [day for day in open_dates if official_date < day <= market_end]
    if registered is not None:
        day = registered["declaration"]["trade_date"]
        previous = [d for d in open_dates if d < day]
        if day != market_end or str(_calendar.get("target_trade_date", "")).replace(
            "-", ""
        ) != day.replace("-", ""):
            raise StrategyRecordStoreError("REGISTERED_CLOSE_HISTORICAL_FINALIZATION_UNSUPPORTED")
        if not previous or previous[-1] != registered["baseline"]["record"]["data_date"]:
            raise StrategyRecordStoreError("REGISTERED_CLOSE_PREVIOUS_OPEN_MISMATCH")
        if pointer != registered["writer"]["pointer"] or official_date != day:
            raise StrategyRecordConflict("REGISTERED_CLOSE_WRITER_CHANGED")
        if any(row["trade_date"] == day for row in event["closures"]):
            raise StrategyRecordConflict("REGISTERED_CLOSE_EMPTY_EVENT_CONFLICT")
        official_date = registered["baseline"]["record"]["data_date"]
        required_candidates = [day]
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
        if expected_plan_sha is not None and len(committed) != 1:
            raise StrategyRecordConflict("expected prepared transaction is not selected")
        if len(committed) == 1:
            transaction_id = str(committed[0].get("transaction_id") or "")
            plan_path = (
                root / "_record_store/daily_close_transactions" / transaction_id / "plan.v1.json"
            )
            if expected_plan_sha is not None:
                _load_json(
                    plan_path, expected_sha=expected_plan_sha, label="expected completed plan"
                )
            if plan_path.exists() and execute:
                plan = json.loads(_read(plan_path, label="daily-close frozen plan"))
                completion_path = _completion_path(root, transaction_id)
                if completion_path.exists():
                    recovered = json.loads(_read(completion_path, label="daily-close completion"))
                    if (
                        recovered.get("schema_id") != BATCH_COMPLETION_SCHEMA
                        or recovered.get("transaction_id") != transaction_id
                        or recovered.get("input_fingerprint") != plan.get("input_fingerprint")
                        or recovered.get("pointer_sha256") != expected_store_pointer_sha
                        or recovered.get("status") not in {"COMMITTED", "RECOVERED_AFTER_CAS"}
                        or recovered.get("content_sha256") != content_sha256(recovered)
                    ):
                        raise StrategyRecordConflict("daily-close completion readback differs")
                else:
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
    active_dir = root / str(pointer["active_record_id"])
    active_ledger = pd.read_parquet(active_dir / "ledger_after_manual_switch.parquet")
    symbols = sorted(active_ledger["symbol"].astype(str).tolist())
    if not symbols or len(symbols) != len(set(symbols)):
        raise StrategyRecordStoreError("active holdings symbol set is invalid")
    table_root = Path(str(market_manifest["table_root"]))
    if not table_root.is_absolute():
        table_root = project / table_root
    partitions: list[Path] = []
    for day in sorted(required_candidates):
        month_part = table_root / f"year={day[:4]}" / f"month={day[5:7]}" / "part.parquet"
        partition = month_part if month_part.is_file() else table_root / "part.parquet"
        if partition not in partitions:
            partitions.append(partition)
    held_symbols = set(symbols)
    held_keys = []
    for partition in partitions:
        try:
            raw = _read(partition, label="held close coverage")
            frame = pd.read_parquet(io.BytesIO(raw), columns=["ts_code", "trade_date", "close"])
            frame = frame.loc[frame["ts_code"].astype(str).isin(held_symbols)]
            for row in frame.to_dict("records"):
                day = str(row.get("trade_date", "")).replace("-", "")
                formatted = f"{day[:4]}-{day[4:6]}-{day[6:]}"
                if formatted in required_candidates:
                    price = float(row.get("close", float("nan")))
                    if math.isfinite(price) and price > 0:
                        held_keys.append((formatted, str(row["ts_code"])))
        except (OSError, ValueError, KeyError, StrategyRecordStoreError):
            continue
    coverage = analyze_close_coverage(
        required_dates=required_candidates,
        event_dates=[row["trade_date"] for row in event["closures"]],
        benchmark_keys=[(row["date"].isoformat(), row["ts_code"]) for row in benchmark["rows"]],
        held_close_keys=held_keys,
        symbols=symbols,
        registered_event_dates=required_candidates if registered is not None else (),
    )
    if coverage["status"] != "READY":
        raise OfficialCloseInputsIncomplete(coverage)
    closures = {row["trade_date"]: row for row in event["closures"]}
    for day in required_candidates:
        if registered is not None:
            continue
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
    profile_fields = (
        _registered_plan_fields(registered, registered_event_declaration_ref)
        if registered is not None
        else {}
    )
    input_fingerprint = _fingerprint(
        {
            "batch_implementation_version": (
                BATCH_IMPLEMENTATION_VERSION if version == 1 else close_contracts.IMPLEMENTATION_V2
            ),
            "requested_target": required_candidates[-1],
            "missing_dates": required_candidates,
            "preimages": preimages,
            **profile_fields,
        }
    )
    transaction_id = (
        f"daily-close-{required_candidates[-1].replace('-', '')}-{input_fingerprint[:16]}"
    )
    transaction_root = root / "_record_store/daily_close_transactions" / transaction_id
    _assert_version_files(root, transaction_id, version)
    plan_path = transaction_root / f"plan.v{version}.json"
    if plan_path.exists():
        plan_raw = _read(plan_path, label="daily-close frozen plan")
        if expected_plan_sha is not None and _sha(plan_raw) != expected_plan_sha:
            raise StrategyRecordConflict("daily-close prepared plan SHA differs")
        plan = json.loads(plan_raw)
        close_contracts.validate_plan(plan, path=plan_path)
        if plan.get("input_fingerprint") != input_fingerprint:
            raise StrategyRecordConflict("daily-close frozen plan input conflict")
    else:
        if expected_plan_sha is not None:
            raise StrategyRecordStoreError("expected daily-close prepared plan is missing")
        planned = (
            (now or datetime.now(timezone.utc)).astimezone(timezone.utc).replace(microsecond=0)
        )
        if registered is not None:
            from quant_investor.strategy_records.registered_event_contracts import stamp

            if planned < stamp(registered["declaration"]["registered_at"]):
                raise StrategyRecordConflict("REGISTERED_CLOSE_PLAN_BEFORE_DECLARATION")
        record_ids = _allocate_record_ids(catalog, planned, len(required_candidates))
        effective_at = planned.isoformat().replace("+00:00", "Z")
        plan = {
            "schema_id": BATCH_PLAN_SCHEMA if version == 1 else close_contracts.PLAN_V2,
            "batch_implementation_version": (
                BATCH_IMPLEMENTATION_VERSION if version == 1 else close_contracts.IMPLEMENTATION_V2
            ),
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
            **profile_fields,
        }
        plan["content_sha256"] = content_sha256(plan)
        close_contracts.validate_plan(plan, path=plan_path)
        if execute or prepare_only:
            _write_exact_json(plan_path, plan)
    if registered is not None:
        _registered_plan_proof(root, plan)
    if execute or prepare_only:
        _retain_close_source_pointer(root, plan)
        if registered is not None:
            _retain_decision_pointer(root, plan)
    if not execute:
        return {
            "status": "PLAN_PREPARED" if prepare_only else "PLAN_READY",
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
        if registered is None:
            closure = closures[day]
            closure_sha = str(closure["content_sha256"])
            event_receipt_id = f"daily-close/{day}/{closure_sha[:16]}"
            receipt_created_at = closure["sealed_at"]
        else:
            continuity = _registered_continuity(plan, registered)
            closure_sha, event_receipt_id = continuity["receipt_sha256"], continuity["receipt_id"]
            receipt_created_at = continuity["receipt_created_at"]
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
                continuity_receipt_created_at=receipt_created_at,
                continuity_checkpoint_digest=content_sha256(source_closure),
                evidence_input_sha256=_sha(canonical_json_bytes(evidences[day])),
                evidence_raw=canonical_json_bytes(evidences[day]),
                publication_class=BATCH_PUBLICATION_CLASS,
                expected_valuation_date=day,
                expected_publication_date=datetime.fromisoformat(
                    plan["effective_at"].replace("Z", "+00:00")
                )
                .astimezone(_SHANGHAI)
                .date()
                .isoformat(),
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
        if registered is not None:
            receipt = _registered_receipt(plan, registered)
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
            **(
                {"official_close_source": registered["writer"]["record"]}
                if registered is not None
                else {}
            ),
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
    owner_sha = immutable_write(
        prefix / "owner_declaration.v1.json",
        owner_raw,
        max_bytes=MAX_PERFORMANCE_JSON_BYTES,
    )
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
    manifest_sha = immutable_write(
        prefix / "manifest.v1.json",
        manifest_raw,
        max_bytes=MAX_PERFORMANCE_JSON_BYTES,
    )
    performance_ref = build_performance_history_ref(
        manifest=performance_manifest,
        manifest_sha256=manifest_sha,
        manifest_bytes=len(manifest_raw),
    )
    load_performance_history(root, performance_ref)
    if registered is not None:
        if _registered_plan_proof(root, plan, retained=True) != registered:
            raise StrategyRecordConflict("REGISTERED_CLOSE_DECLARATION_CHANGED")
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
