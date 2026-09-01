#!/usr/bin/env python3
"""Prepare or publish the derived-only CN strategy accounting genesis.

The command is offline.  It reads one exact Store-v3 head, bounded archived
members through their registered locator/manifest/restore receipt, and one
explicit SW2021 membership capture.  Historical material is used only to seal
a gap audit; prospective opening lots come exclusively from the current active
Parquet ledger.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import secrets
import sys
import tarfile
from typing import Any, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_investor.strategy_records.accounting import (  # noqa: E402
    ACCOUNTING_POINTER_SCHEMA,
    HISTORICAL_GAP_AUDIT_SCHEMA,
    StrategyAccountingError,
    build_genesis,
    immutable_write,
    load_accounting_generation,
    seal_document,
    validate_genesis,
)
from quant_investor.strategy_records.performance import load_performance_history  # noqa: E402
from quant_investor.strategy_records.store import (  # noqa: E402
    canonical_json_bytes,
    load_archive_binding,
    load_registered_catalog,
)

_MAX_ARCHIVE_DECOMPRESSED = 512 * 1024 * 1024
_MAX_MEMBER = 64 * 1024 * 1024


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read(path: Path, *, label: str, max_bytes: int = 64 * 1024 * 1024) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise StrategyAccountingError(f"{label} is not a regular file")
    if path.stat().st_size > max_bytes:
        raise StrategyAccountingError(f"{label} exceeds byte bound")
    first = path.read_bytes()
    if first != path.read_bytes():
        raise StrategyAccountingError(f"{label} changed during read")
    return first


def _json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _read(path, label=label)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StrategyAccountingError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise StrategyAccountingError(f"{label} is not an object")
    return value, raw


class RecordReader:
    """Bounded exact-member reader for registered ONLINE/ARCHIVED records."""

    def __init__(self, *, project: Path, record_root: Path, records: list[dict[str, Any]]):
        self.project = project
        self.record_root = record_root
        self.by_id = {str(row["record_id"]): row for row in records}
        self.archives: dict[str, tuple[tarfile.TarFile, dict[str, tarfile.TarInfo]]] = {}
        self.source_refs: dict[str, str] = {}

    def _archive(
        self, record: Mapping[str, Any]
    ) -> tuple[tarfile.TarFile, dict[str, tarfile.TarInfo]]:
        loaded = load_archive_binding(
            self.record_root,
            record,
            project_root=self.project,
        )
        locator = record["archive_locator"]
        archive_id = str(locator["archive_id"])
        for key in ("archive_path", "manifest_path", "restore_receipt_path"):
            self.source_refs[str(locator[key])] = str(
                locator[
                    (
                        "archive_sha256"
                        if key == "archive_path"
                        else (
                            "manifest_sha256"
                            if key == "manifest_path"
                            else "restore_receipt_sha256"
                        )
                    )
                ]
            )
        if archive_id in self.archives:
            return self.archives[archive_id]
        try:
            completed = subprocess.run(
                ["zstd", "-dc", str(loaded["archive_path"])],
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise StrategyAccountingError("registered archive could not be decoded") from exc
        if completed.returncode != 0 or len(completed.stdout) > _MAX_ARCHIVE_DECOMPRESSED:
            raise StrategyAccountingError("registered archive decode failed or exceeded bound")
        try:
            handle = tarfile.open(fileobj=io.BytesIO(completed.stdout), mode="r:")
        except tarfile.TarError as exc:
            raise StrategyAccountingError("registered archive tar is invalid") from exc
        members: dict[str, tarfile.TarInfo] = {}
        for member in handle.getmembers():
            if member.isfile():
                if member.size > _MAX_MEMBER or member.name in members:
                    raise StrategyAccountingError("registered archive member is unsafe")
                members[member.name] = member
        self.archives[archive_id] = (handle, members)
        return handle, members

    def read(self, record_id: str, filename: str) -> tuple[bytes, str] | None:
        record = self.by_id.get(record_id)
        if record is None:
            raise StrategyAccountingError(f"record is not catalog registered:{record_id}")
        inventory = {
            str(row.get("path")): row
            for row in record.get("inventory", [])
            if isinstance(row, dict) and row.get("type") == "file"
        }
        expected = inventory.get(filename)
        if expected is None:
            return None
        if record.get("storage_state") == "ONLINE":
            relative = Path(str(record["relative_path"])) / filename
            path = self.record_root / relative
            raw = _read(path, label="online record member", max_bytes=_MAX_MEMBER)
            logical = path.relative_to(self.project).as_posix()
        elif record.get("storage_state") == "ARCHIVED":
            handle, members = self._archive(record)
            prefix = str(record["archive_locator"]["member_prefix"])
            candidates = [
                name
                for name in members
                if name == f"{prefix}/{filename}" or name.endswith(f"/{prefix}/{filename}")
            ]
            if len(candidates) != 1:
                raise StrategyAccountingError("archive member identity is ambiguous")
            extracted = handle.extractfile(members[candidates[0]])
            if extracted is None:
                raise StrategyAccountingError("archive member cannot be read")
            raw = extracted.read(_MAX_MEMBER + 1)
            logical = (
                self.record_root.relative_to(self.project) / str(record["relative_path"]) / filename
            ).as_posix()
        else:
            raise StrategyAccountingError("record storage state is unsupported")
        if len(raw) != int(expected["size"]) or _sha(raw) != expected["sha256"]:
            raise StrategyAccountingError("record member differs from catalog inventory")
        self.source_refs[logical] = str(expected["sha256"])
        return raw, logical


def _side(value: Any) -> str | None:
    text = str(value or "").upper()
    if "SELL" in text:
        return "SELL"
    if "BUY" in text or "ADD" in text:
        return "BUY"
    return None


def _trade_date(value: Any, *, fallback: str) -> str:
    raw = str(value or fallback).replace("-", "")[:8]
    if len(raw) != 8 or not raw.isdigit():
        return fallback
    return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"


def _normalize_trade(
    row: Mapping[str, Any],
    *,
    record_id: str,
    index: int,
    source_path: str,
    source_sha: str,
    fallback_date: str,
) -> dict[str, Any] | None:
    side = _side(row.get("side") or row.get("action"))
    symbol = row.get("symbol")
    try:
        shares = int(float(row.get("shares") or 0))
        price = Decimal(str(row.get("execution_price") or row.get("price") or 0))
    except (ValueError, ArithmeticError):
        return None
    if side is None or not isinstance(symbol, str) or shares <= 0 or price <= 0:
        return None
    provided_id = row.get("trade_id")
    event_id = (
        str(provided_id)
        if isinstance(provided_id, str) and provided_id
        else "legacy-"
        + _sha(
            canonical_json_bytes(
                {
                    "record_id": record_id,
                    "index": index,
                    "symbol": symbol,
                    "side": side,
                    "shares": shares,
                    "price": str(price),
                    "source_sha": source_sha,
                }
            )
        )[:24]
    )
    known_fee = row.get("final_total_fee_cny")
    fee_status = "KNOWN" if known_fee is not None else "LEGACY_UNAVAILABLE"
    return {
        "event_id": event_id,
        "record_id": record_id,
        "trade_date": _trade_date(row.get("trade_date"), fallback=fallback_date),
        "symbol": symbol,
        "name": str(row.get("name") or ""),
        "side": side,
        "shares": shares,
        "price_cny": format(price, "f"),
        "gross_amount_cny": format(price * shares, "f"),
        "fee_status": fee_status,
        "total_fee_cny": format(Decimal(str(known_fee)), "f") if known_fee is not None else None,
        "reported_realized_pnl_cny": (
            format(Decimal(str(row["realized_pnl"])), "f")
            if row.get("realized_pnl") is not None
            else None
        ),
        "source_path": source_path,
        "source_sha256": source_sha,
        "authority": (
            "OWNER_DECLARED_APPLIED_FILL"
            if "owner" in str(row.get("status") or row.get("execution_origin") or "").lower()
            or row.get("user_reported") is True
            else "LEGACY_LINEAGE_REPORTED_FILL"
        ),
        "prospective_authority": False,
    }


def _ledger_rows(raw: bytes, *, suffix: str) -> list[dict[str, Any]]:
    if suffix == ".parquet":
        return pd.read_parquet(io.BytesIO(raw)).to_dict("records")
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))


def _extract_historical_audit(
    *,
    reader: RecordReader,
    catalog: dict[str, Any],
    performance: dict[str, Any],
    final_positions: list[dict[str, Any]],
    source_store: dict[str, Any],
    created_at: str,
) -> dict[str, Any]:
    first_record = str(performance["rows"][0]["record_id"])
    opening_member = reader.read(first_record, "ledger.csv")
    if opening_member is None:
        raise StrategyAccountingError("historical opening ledger is unavailable")
    opening_raw, opening_path = opening_member
    opening_rows = _ledger_rows(opening_raw, suffix=".csv")
    opening_shares = {
        str(row["symbol"]): int(float(row["shares"]))
        for row in opening_rows
        if row.get("symbol") and row.get("shares")
    }
    calculated = dict(opening_shares)
    trades: list[dict[str, Any]] = []
    unknown_fee_count = 0
    known_fee_count = 0
    for lineage in catalog["lineage_index"]:
        record_id = str(lineage["record_id"])
        manifest_member = reader.read(record_id, "manifest.json")
        if manifest_member is None:
            continue
        manifest_raw, manifest_path = manifest_member
        manifest = json.loads(manifest_raw)
        manual_member = reader.read(record_id, "manual_execution_manifest.json")
        if manual_member is not None:
            manual_raw, source_path = manual_member
            manual = json.loads(manual_raw)
            source_sha = _sha(manual_raw)
        else:
            manual = manifest.get("manual_execution") or {}
            source_path = manifest_path
            source_sha = _sha(manifest_raw)
        applied = list(manual.get("applied_owner_declared_trades") or []) + list(
            manual.get("applied_local_trades") or []
        )
        source_rows: list[Mapping[str, Any]] = applied
        if not source_rows and manifest.get("action_taken_today") is True:
            orders_member = reader.read(record_id, "orders.csv")
            if orders_member is not None:
                orders_raw, source_path = orders_member
                source_sha = _sha(orders_raw)
                source_rows = list(csv.DictReader(io.StringIO(orders_raw.decode("utf-8-sig"))))
        fallback_date = str(lineage.get("valuation_date") or "")
        for index, raw in enumerate(source_rows, start=1):
            trade = _normalize_trade(
                raw,
                record_id=record_id,
                index=index,
                source_path=source_path,
                source_sha=source_sha,
                fallback_date=fallback_date,
            )
            if trade is None:
                continue
            calculated[trade["symbol"]] = calculated.get(trade["symbol"], 0) + (
                trade["shares"] if trade["side"] == "BUY" else -trade["shares"]
            )
            if trade["fee_status"] == "KNOWN":
                known_fee_count += 1
            else:
                unknown_fee_count += 1
            trades.append(trade)
    final_shares = {str(row["symbol"]): int(row["shares"]) for row in final_positions}
    unexplained = [
        {
            "symbol": symbol,
            "reported_path_shares": calculated.get(symbol, 0),
            "current_store_shares": final_shares.get(symbol, 0),
            "unexplained_delta": final_shares.get(symbol, 0) - calculated.get(symbol, 0),
        }
        for symbol in sorted(set(calculated) | set(final_shares))
        if calculated.get(symbol, 0) != final_shares.get(symbol, 0)
    ]
    return seal_document(
        {
            "schema_id": HISTORICAL_GAP_AUDIT_SCHEMA,
            "created_at": created_at,
            "strategy_label": "aggressive_tech_manufacturing",
            "historical_start_record_id": first_record,
            "historical_opening_ledger_ref": {
                "path": opening_path,
                "sha256": _sha(opening_raw),
            },
            "current_active_record_id": source_store["active_record_id"],
            "reported_fill_count": len(trades),
            "known_fee_fill_count": known_fee_count,
            "legacy_unavailable_fee_fill_count": unknown_fee_count,
            "reported_fills": trades,
            "unexplained_share_deltas": unexplained,
            "status": "HISTORICAL_PARTIAL",
            "prospective_lot_authority": False,
            "blockers": [
                "HISTORICAL_SHARE_TRANSITIONS_UNEXPLAINED",
                "LEGACY_FEES_UNAVAILABLE",
                "HISTORICAL_REALIZED_PNL_UNALLOCATED",
            ],
        }
    )


def _industry_rows(
    *,
    capture_path: Path,
    expected_capture_sha: str,
    symbols: set[str],
    effective_date: str,
    project: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    capture, raw = _json(capture_path, label="SW2021 membership capture")
    if _sha(raw) != expected_capture_sha or capture.get("status") != "COMPLETE":
        raise StrategyAccountingError("SW2021 membership capture is not exact COMPLETE")
    base = capture_path.parent
    matches: dict[str, list[dict[str, Any]]] = {symbol: [] for symbol in symbols}
    selected_refs: dict[str, str] = {
        capture_path.relative_to(project).as_posix(): expected_capture_sha
    }
    compact = effective_date.replace("-", "")
    for ref in capture.get("partition_rows", []):
        if not isinstance(ref, dict):
            raise StrategyAccountingError("SW2021 partition ref is invalid")
        path = base / str(ref.get("relative_path"))
        partition_raw = _read(path, label="SW2021 membership partition")
        if _sha(partition_raw) != ref.get("byte_sha256"):
            raise StrategyAccountingError("SW2021 partition SHA differs")
        partition = json.loads(partition_raw)
        relevant = False
        for row in partition.get("rows", []):
            symbol = row.get("ts_code")
            if symbol not in matches:
                continue
            in_date = str(row.get("in_date") or "")
            out_date = row.get("out_date")
            if in_date <= compact and (out_date is None or compact <= str(out_date)):
                matches[symbol].append(dict(row))
                relevant = True
        if relevant:
            selected_refs[path.relative_to(project).as_posix()] = str(ref["byte_sha256"])
    industry: list[dict[str, Any]] = []
    for symbol in sorted(symbols):
        rows = matches[symbol]
        if len(rows) != 1:
            raise StrategyAccountingError(f"SW2021 PIT membership is missing/ambiguous:{symbol}")
        row = rows[0]
        industry.append(
            {
                "symbol": symbol,
                "valid_from": f"{row['in_date'][:4]}-{row['in_date'][4:6]}-{row['in_date'][6:]}",
                "valid_to": (
                    f"{row['out_date'][:4]}-{row['out_date'][4:6]}-{row['out_date'][6:]}"
                    if row.get("out_date")
                    else None
                ),
                "industry_l1_code": row["l1_code"],
                "industry_l1_name": row["l1_name"],
                "industry_l2_code": row["l2_code"],
                "industry_l2_name": row["l2_name"],
                "industry_l3_code": row["l3_code"],
                "industry_l3_name": row["l3_name"],
                "classification_source": "TUSHARE_SW2021",
                "taxonomy_version": "SW2021",
                "available_at": capture["timestamp"],
            }
        )
    refs = [{"path": path, "sha256": selected_refs[path]} for path in sorted(selected_refs)]
    return industry, refs


def _pointer_sha(path: Path) -> str | None:
    return _sha(_read(path, label="accounting pointer")) if path.exists() else None


def _publish_pointer(path: Path, document: dict[str, Any], *, expected: str | None) -> str:
    raw = canonical_json_bytes(document)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = path.parent / ".current.v1.lock"
    lock_fd = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        if os.fstat(lock_fd).st_nlink != 1:
            raise StrategyAccountingError("accounting pointer lock is unsafe")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        current_raw = _read(path, label="accounting pointer") if path.exists() else None
        observed = _sha(current_raw) if current_raw is not None else None
        if observed != expected:
            raise StrategyAccountingError("accounting pointer preimage mismatch")
        if current_raw == raw:
            return observed or _sha(raw)
        if current_raw is not None:
            immutable_write(
                path.parent / "pointer_history" / f"{observed}.json",
                current_raw,
                max_bytes=1024 * 1024,
            )
        temporary = path.parent / (f".current.v1.cas-{os.getpid()}-{secrets.token_hex(6)}")
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            os.write(fd, raw)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if _read(path, label="accounting pointer readback") != raw:
            raise StrategyAccountingError("accounting pointer readback differs")
        return _sha(raw)
    finally:
        os.close(lock_fd)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    project = Path(args.project_root).resolve(strict=True)
    record_root = Path(args.record_root).resolve(strict=True)
    pointer_path = record_root / "_record_store/current.v1.json"
    pointer, pointer_raw = _json(pointer_path, label="Store pointer")
    pointer_sha = _sha(pointer_raw)
    if pointer_sha != args.expected_store_pointer_sha:
        raise StrategyAccountingError("Store pointer preimage mismatch")
    loaded = load_registered_catalog(record_root)
    if loaded is None:
        raise StrategyAccountingError("Store-v3 is unregistered")
    loaded_pointer, catalog = loaded
    if loaded_pointer != pointer:
        raise StrategyAccountingError("Store pointer changed during load")
    catalog_path = record_root / pointer["catalog_path"]
    catalog_raw = _read(catalog_path, label="Store catalog")
    if _sha(catalog_raw) != pointer["catalog_sha256"]:
        raise StrategyAccountingError("Store catalog SHA differs")
    performance = load_performance_history(record_root, catalog["performance_history_ref"])
    manifest_ref = catalog["performance_history_ref"]["manifest"]
    series_ref = catalog["performance_history_ref"]["series"]
    active = record_root / pointer["active_record_id"]
    ledger_path = active / "ledger_after_manual_switch.parquet"
    ledger_raw = _read(ledger_path, label="active Parquet ledger")
    if _sha(ledger_raw) != pointer["active_closure"]["ledger_sha256"]:
        raise StrategyAccountingError("active ledger SHA differs")
    table = pd.read_parquet(io.BytesIO(ledger_raw))
    positions = [
        {
            "symbol": str(row.symbol),
            "name": str(row.name),
            "shares": int(row.shares),
            "avg_cost_cny": str(row.avg_cost),
            "cost_basis_cny": str(row.cost_basis),
            "market_value_cny": str(row.current_value),
        }
        for row in table.itertuples()
    ]
    final = performance["rows"][-1]
    effective_date = str(final["valuation_date"])
    if "price_date" in table and effective_date != str(table["price_date"].iloc[0]):
        raise StrategyAccountingError("active ledger price date differs")
    source_store = {
        "pointer_sha256": pointer_sha,
        "catalog_generation_id": pointer["generation_id"],
        "catalog_sha256": pointer["catalog_sha256"],
        "performance_generation_id": performance["manifest"]["performance_generation_id"],
        "performance_manifest_sha256": manifest_ref["sha256"],
        "performance_series_sha256": series_ref["sha256"],
        "active_record_id": pointer["active_record_id"],
        "active_ledger_sha256": pointer["active_closure"]["ledger_sha256"],
    }
    reader = RecordReader(project=project, record_root=record_root, records=catalog["records"])
    preimages = {
        "source_store": source_store,
        "industry_capture_sha256": args.industry_capture_sha,
    }
    fingerprint = _sha(canonical_json_bytes(preimages))
    generation_id = f"accounting-cutover-{effective_date.replace('-', '')}-{fingerprint[:16]}"
    accounting_root = record_root / "_accounting_store"
    transaction_root = accounting_root / "transactions" / fingerprint
    plan_path = transaction_root / "plan.v1.json"
    if plan_path.exists():
        plan, _ = _json(plan_path, label="accounting cutover plan")
        if plan.get("input_fingerprint") != fingerprint:
            raise StrategyAccountingError("accounting plan input conflicts")
    else:
        created_at = (
            datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        plan = seal_document(
            {
                "schema_id": "myquant.strategy_accounting_cutover_plan.v1",
                "input_fingerprint": fingerprint,
                "generation_id": generation_id,
                "created_at": created_at,
                "effective_date": effective_date,
                "source_store_pointer_sha256": pointer_sha,
                "derived_only": True,
                "broker_order_trade_authority": False,
            }
        )
        if args.execute:
            immutable_write(plan_path, canonical_json_bytes(plan))
    audit = _extract_historical_audit(
        reader=reader,
        catalog=catalog,
        performance=performance,
        final_positions=positions,
        source_store=source_store,
        created_at=plan["created_at"],
    )
    industry, industry_refs = _industry_rows(
        capture_path=project / args.industry_capture,
        expected_capture_sha=args.industry_capture_sha,
        symbols={row["symbol"] for row in positions},
        effective_date=effective_date,
        project=project,
    )
    source_refs = {
        pointer_path.relative_to(project).as_posix(): pointer_sha,
        catalog_path.relative_to(project).as_posix(): pointer["catalog_sha256"],
        (record_root / manifest_ref["path"])
        .relative_to(project)
        .as_posix(): manifest_ref["sha256"],
        (record_root / series_ref["path"]).relative_to(project).as_posix(): series_ref["sha256"],
        ledger_path.relative_to(project).as_posix(): source_store["active_ledger_sha256"],
        **reader.source_refs,
        **{row["path"]: row["sha256"] for row in industry_refs},
    }
    refs = [{"path": path, "sha256": source_refs[path]} for path in sorted(source_refs)]
    theme = [
        {
            "symbol": row["symbol"],
            "theme_id": "OTHER_UNCLASSIFIED",
            "weight": "1",
            "confidence": "UNVERIFIED",
            "classification_kind": "PRIMARY_THEME_ATTRIBUTION_BUCKET",
            "economic_exposure_claimed": False,
            "basis": "NO_COMPANY_ECONOMIC_EXPOSURE_CLOSURE",
        }
        for row in positions
    ]
    generation_dir = accounting_root / "generations" / generation_id
    audit_path = generation_dir / "historical-gap-audit.v1.json"
    audit_sha = _sha(canonical_json_bytes(audit))
    genesis = build_genesis(
        generation_id=generation_id,
        created_at=plan["created_at"],
        strategy_label="aggressive_tech_manufacturing",
        effective_date=effective_date,
        source_store=source_store,
        source_refs=refs,
        cash_cny=final["cash_cny"],
        nav_cny=final["raw_nav_cny"],
        portfolio_pnl_cny=final["portfolio_pnl_cny"],
        positions=positions,
        historical_audit_ref={
            "path": audit_path.relative_to(record_root).as_posix(),
            "sha256": audit_sha,
        },
        industry_rows=industry,
        theme_rows=theme,
    )
    validate_genesis(genesis)
    if not args.execute:
        return {
            "status": "PLAN_READY",
            "generation_id": generation_id,
            "effective_date": effective_date,
            "historical_status": "PARTIAL",
            "prospective_status": "READY",
            "reported_fill_count": audit["reported_fill_count"],
            "unexplained_share_delta_count": len(audit["unexplained_share_deltas"]),
            "provider_calls": False,
            "broker_calls": False,
            "order_calls": False,
            "trade_calls": False,
        }
    immutable_write(audit_path, canonical_json_bytes(audit))
    genesis_path = generation_dir / "genesis.v1.json"
    genesis_sha = immutable_write(genesis_path, canonical_json_bytes(genesis))
    if _sha(_read(pointer_path, label="Store pointer final preimage")) != pointer_sha:
        raise StrategyAccountingError("Store pointer drifted before accounting CAS")
    accounting_pointer_path = accounting_root / "current.v1.json"
    expected_accounting = (
        None
        if args.expected_accounting_pointer_sha == "ABSENT"
        else args.expected_accounting_pointer_sha
    )
    accounting_pointer = seal_document(
        {
            "schema_id": ACCOUNTING_POINTER_SCHEMA,
            "generation_id": generation_id,
            "genesis_path": genesis_path.relative_to(record_root).as_posix(),
            "genesis_sha256": genesis_sha,
            "historical_audit_sha256": audit_sha,
            "source_store_pointer_sha256": pointer_sha,
            "published_at": plan["created_at"],
            "derived_only": True,
            "store_mutation_authority": False,
            "holdings_mutation_authority": False,
            "broker_order_trade_authority": False,
        }
    )
    before = _pointer_sha(accounting_pointer_path)
    if before is not None:
        existing, _ = _json(accounting_pointer_path, label="accounting pointer")
        if existing == accounting_pointer and before == expected_accounting:
            return {
                "status": "NO_ACTION",
                "generation_id": generation_id,
                "pointer_sha256": before,
                "historical_status": "PARTIAL",
                "prospective_status": "READY",
                "provider_calls": False,
                "broker_calls": False,
                "order_calls": False,
                "trade_calls": False,
            }
    pointer_digest = _publish_pointer(
        accounting_pointer_path,
        accounting_pointer,
        expected=expected_accounting,
    )
    return {
        "status": "PUBLISHED",
        "generation_id": generation_id,
        "pointer_sha256": pointer_digest,
        "historical_status": "PARTIAL",
        "prospective_status": "READY",
        "reported_fill_count": audit["reported_fill_count"],
        "unexplained_share_delta_count": len(audit["unexplained_share_deltas"]),
        "provider_calls": False,
        "broker_calls": False,
        "order_calls": False,
        "trade_calls": False,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--project-root", default=str(ROOT))
    result.add_argument(
        "--record-root",
        default=str(ROOT / "results/strategy_records/CN/aggressive_tech_manufacturing"),
    )
    result.add_argument("--expected-store-pointer-sha")
    result.add_argument(
        "--industry-capture",
        default=(
            "data/private/intelligence_sources/industry/sw2021/"
            "sw2021-industry-20260822-563a6794c22c/membership-capture/capture.json"
        ),
    )
    result.add_argument("--industry-capture-sha")
    result.add_argument("--expected-accounting-pointer-sha", default="ABSENT")
    result.add_argument("--execute", action="store_true")
    result.add_argument("--verify", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if args.verify:
            verified = load_accounting_generation(Path(args.record_root))
            output = {
                "status": verified["state"],
                "generation_id": verified["pointer"]["generation_id"],
                "pointer_sha256": verified["pointer_sha256"],
                "source_store_pointer_sha256": verified["source_store_pointer_sha256"],
                "current_store_pointer_sha256": verified["current_store_pointer_sha256"],
                "coverage": verified["genesis"]["coverage"],
                "accounting_status": verified["genesis"]["status"],
                "reported_fill_count": verified["audit"]["reported_fill_count"],
                "unexplained_share_delta_count": len(verified["audit"]["unexplained_share_deltas"]),
                "provider_calls": False,
                "broker_calls": False,
                "order_calls": False,
                "trade_calls": False,
            }
        else:
            output = prepare(args)
    except (StrategyAccountingError, OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, **output}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
