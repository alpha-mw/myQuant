"""Bounded CN market maintenance state machine.

This module stages raw CN maintenance work into resumable batches. Only a
downloader that explicitly advertises a non-canonical-safe sink may execute;
otherwise the run is blocked before downloader/provider construction. Partial
maintenance results must never promote the Parquet canonical pointer.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from quant_investor.market.config import get_market_settings
from quant_investor.market.download_cn import CNFullMarketDownloader

SUCCESS_STATUSES = frozenset({"updated", "cached", "stale_cached"})
RUNNABLE_BATCH_STATUSES = frozenset({"pending", "running", "incomplete", "failed"})
COMPLETED_BATCH_STATUS = "completed"
SCHEMA_VERSION = "myquant-cn-staged-maintenance.v1"
NONCANONICAL_WRITER_BLOCKER = "blocked_noncanonical_writer_disabled"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _compact_trade_date(value: Any) -> str:
    text = str(value or "").strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else ""


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _maintenance_root(data_dir: str | Path) -> Path:
    return Path(data_dir).expanduser() / "_maintenance_runs"


def _write_noncanonical_writer_block(
    *,
    data_dir: str | Path,
    categories: list[str] | None,
    target_date: str,
    resume: bool,
    storage_validate: dict[str, Any] | None,
) -> Path:
    requested_target = _compact_trade_date(target_date)
    effective_target = requested_target or "unknown"
    run_id = _new_run_id(effective_target)
    run_dir = _maintenance_root(data_dir) / run_id
    progress_path = run_dir / "progress_summary.json"
    target_categories = _normalize_categories(categories) or ["full_a"]
    generated_at = _utc_now_iso()
    progress = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "run_id": run_id,
        "market": "CN",
        "categories": target_categories,
        "status": "blocked",
        "maintenance_status": "blocked",
        "complete": False,
        "decision_data_sufficient": False,
        "decision_data_status": "blocked",
        "target_trade_date": requested_target or str(target_date or "auto"),
        "effective_target_trade_date": requested_target,
        "resume_requested": bool(resume),
        "early_stop_reason": NONCANONICAL_WRITER_BLOCKER,
        "limitations": [NONCANONICAL_WRITER_BLOCKER],
        "blockers": [NONCANONICAL_WRITER_BLOCKER],
        "storage_validate": dict(storage_validate or {}),
        "run_dir": str(run_dir),
        "progress_summary_path": str(progress_path),
    }
    _atomic_json_write(progress_path, progress)
    return progress_path


def _canonical_pointer_state(downloader: Any) -> dict[str, Any]:
    reader = getattr(downloader, "market_reader", None)
    pointer_path = getattr(reader, "latest_pointer_path", None)
    if pointer_path is None:
        data_root = getattr(downloader, "data_root", None)
        if data_root:
            pointer_path = Path(data_root) / "parquet" / "cn" / "_latest.json"
    if pointer_path is None:
        return {"path": "", "exists": False, "sha256": ""}
    path = Path(pointer_path)
    if not path.exists():
        return {"path": str(path), "exists": False, "sha256": ""}
    payload = path.read_bytes()
    return {
        "path": str(path),
        "exists": True,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _normalize_categories(categories: Iterable[str] | None) -> list[str]:
    normalized: list[str] = []
    for category in categories or []:
        text = str(category or "").strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _resolve_effective_target(
    *,
    downloader: Any,
    target_date: str,
    same_day_probe: dict[str, Any],
) -> tuple[str, str, str]:
    explicit = _compact_trade_date(target_date)
    if explicit and str(target_date or "").strip().lower() != "auto":
        return explicit, explicit, ""

    strict_target = _compact_trade_date(
        getattr(downloader, "strict_trade_date", "")
        or getattr(downloader, "latest_trade_date", "")
    )
    effective = _compact_trade_date(getattr(downloader, "latest_trade_date", "") or strict_target)
    early_stop_reason = ""
    if same_day_probe.get("applicable") and same_day_probe.get("available") is False:
        stable = _compact_trade_date(getattr(downloader, "stable_trade_date", ""))
        if stable:
            effective = stable
            early_stop_reason = "strict_same_day_unavailable"
    return strict_target or effective, effective, early_stop_reason


def _apply_daily_window(downloader: Any, effective_target_trade_date: str) -> None:
    target = _compact_trade_date(effective_target_trade_date)
    if not target:
        return
    try:
        target_dt = datetime.strptime(target, "%Y%m%d")
    except ValueError:
        return
    downloader.start_date = target_dt - timedelta(days=10)
    downloader.end_date = target_dt


def _collect_blocking_by_category(downloader: Any, completeness: dict[str, Any], categories: list[str]) -> dict[str, list[str]]:
    blocking: dict[str, list[str]] = {}
    payloads = completeness.get("categories") if isinstance(completeness, dict) else {}
    if not isinstance(payloads, dict):
        return blocking
    seen_symbols: set[str] = set()
    for category in categories:
        payload = payloads.get(category, {})
        if hasattr(downloader, "_collect_blocking_symbols"):
            raw_symbols = downloader._collect_blocking_symbols(payload)
        else:
            raw_symbols = list(payload.get("blocking_missing_symbols", []) or [])
        deduped: list[str] = []
        for symbol in raw_symbols or []:
            text = str(symbol or "").strip().upper()
            if text and text not in seen_symbols and text not in deduped:
                deduped.append(text)
                seen_symbols.add(text)
        if deduped:
            blocking[category] = deduped
    return blocking


def _dedupe_batch_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped_rows: list[dict[str, Any]] = []
    seen_symbols: set[str] = set()
    for row in rows:
        symbols: list[str] = []
        for symbol in row.get("symbols") or []:
            text = str(symbol or "").strip().upper()
            if text and text not in seen_symbols:
                symbols.append(text)
                seen_symbols.add(text)
        if not symbols:
            continue
        next_row = {**row, "symbols": symbols, "symbol_count": len(symbols)}
        completed = min(int(next_row.get("completed_symbol_count") or 0), len(symbols))
        failed = min(int(next_row.get("failed_symbol_count") or 0), len(symbols))
        next_row["completed_symbol_count"] = completed
        next_row["failed_symbol_count"] = failed
        deduped_rows.append(next_row)
    return deduped_rows


def _build_batches(blocking_by_category: dict[str, list[str]], batch_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bounded_batch_size = max(1, int(batch_size or 200))
    batch_no = 1
    for category in sorted(blocking_by_category):
        symbols = blocking_by_category[category]
        for start in range(0, len(symbols), bounded_batch_size):
            chunk = symbols[start : start + bounded_batch_size]
            rows.append(
                {
                    "batch_id": f"{batch_no:04d}",
                    "category": category,
                    "status": "pending",
                    "symbols": list(chunk),
                    "symbol_count": len(chunk),
                    "completed_symbol_count": 0,
                    "failed_symbol_count": 0,
                    "updated_at": _utc_now_iso(),
                    "manifest_path": "",
                }
            )
            batch_no += 1
    return rows


def _write_batches_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now_iso(),
        "batches": rows,
    }
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_batches_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_rows = payload.get("batches") if isinstance(payload, dict) else payload
    rows: list[dict[str, Any]] = []
    for row in raw_rows or []:
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "batch_id": str(row.get("batch_id") or ""),
                "category": str(row.get("category") or ""),
                "status": str(row.get("status") or "pending"),
                "symbols": [
                    str(symbol or "").strip().upper()
                    for symbol in list(row.get("symbols") or [])
                    if str(symbol or "").strip()
                ],
                "symbol_count": int(row.get("symbol_count") or 0),
                "completed_symbol_count": int(row.get("completed_symbol_count") or 0),
                "failed_symbol_count": int(row.get("failed_symbol_count") or 0),
                "updated_at": str(row.get("updated_at") or ""),
                "manifest_path": str(row.get("manifest_path") or ""),
            }
        )
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _latest_resumable_run(
    *,
    root: Path,
    categories: list[str],
    effective_target_trade_date: str,
) -> Path | None:
    if not root.exists():
        return None
    category_key = sorted(categories)
    candidates: list[Path] = []
    for progress_path in root.glob("*/progress_summary.json"):
        payload = _read_json(progress_path)
        if str(payload.get("status") or "") == "complete":
            continue
        if _compact_trade_date(payload.get("effective_target_trade_date")) != effective_target_trade_date:
            continue
        if sorted(payload.get("categories") or []) != category_key:
            continue
        candidates.append(progress_path.parent)
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _new_run_id(effective_target_trade_date: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = _compact_trade_date(effective_target_trade_date) or "unknown"
    return f"cn_staged_{target}_{stamp}_{uuid.uuid4().hex[:8]}"


def _batch_manifest_path(run_dir: Path, batch_id: str) -> Path:
    return run_dir / f"batch_{batch_id}" / "manifest.json"


def _write_progress(
    *,
    run_dir: Path,
    run_id: str,
    categories: list[str],
    batch_size: int,
    max_batches_per_run: int,
    target_trade_date: str,
    effective_target_trade_date: str,
    min_symbol_success_rate: float,
    completeness: dict[str, Any],
    batches: list[dict[str, Any]],
    same_day_probe: dict[str, Any],
    early_stop_reason: str,
    storage_validate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    completed_batches = [row for row in batches if row.get("status") == COMPLETED_BATCH_STATUS]
    remaining_batches = [row for row in batches if row.get("status") != COMPLETED_BATCH_STATUS]
    successful_symbols: list[str] = []
    failed_symbols: list[str] = []
    failed_batches: list[dict[str, Any]] = []
    for row in batches:
        manifest_path = Path(str(row.get("manifest_path") or _batch_manifest_path(run_dir, row["batch_id"])))
        manifest = _read_json(manifest_path)
        for symbol in manifest.get("successful_symbols") or []:
            if symbol not in successful_symbols:
                successful_symbols.append(symbol)
        batch_failed_symbols = [str(symbol) for symbol in manifest.get("failed_symbols") or [] if str(symbol).strip()]
        for symbol in batch_failed_symbols:
            if symbol not in failed_symbols:
                failed_symbols.append(symbol)
        if batch_failed_symbols:
            failed_batches.append(
                {
                    "batch_id": row.get("batch_id"),
                    "category": row.get("category"),
                    "status": row.get("status"),
                    "failed_symbols": batch_failed_symbols,
                    "manifest_path": str(manifest_path),
                }
            )

    coverage_ratio = float(completeness.get("coverage_ratio") or 0.0)
    complete = bool(completeness.get("complete"))
    decision_data_sufficient = bool(complete or coverage_ratio >= float(min_symbol_success_rate))
    if complete:
        status = "complete"
        decision_status = "sufficient"
    elif decision_data_sufficient:
        status = "running" if completed_batches and remaining_batches else "incomplete"
        decision_status = "sufficient_limited"
    else:
        status = "running" if completed_batches and remaining_batches else "incomplete"
        decision_status = "limited"

    progress = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now_iso(),
        "run_id": run_id,
        "market": "CN",
        "categories": list(categories),
        "status": status,
        "maintenance_status": status,
        "complete": complete,
        "total_symbols": sum(int(row.get("symbol_count") or len(row.get("symbols") or [])) for row in batches),
        "batch_size": int(batch_size),
        "max_batches_per_run": int(max_batches_per_run),
        "completed_batches": len(completed_batches),
        "remaining_batches": len(remaining_batches),
        "completed_symbols": len(successful_symbols),
        "failed_symbols": failed_symbols,
        "failed_symbol_count": len(failed_symbols),
        "target_trade_date": target_trade_date,
        "effective_target_trade_date": effective_target_trade_date,
        "strict_trade_date": completeness.get("strict_trade_date"),
        "stable_trade_date": completeness.get("stable_trade_date"),
        "coverage_ratio": coverage_ratio,
        "blocking_incomplete_count": int(completeness.get("blocking_incomplete_count") or 0),
        "decision_data_sufficient": decision_data_sufficient,
        "decision_data_status": decision_status,
        "min_symbol_success_rate": float(min_symbol_success_rate),
        "same_day_close_probe": same_day_probe,
        "early_stop_reason": early_stop_reason,
        "limitations": [] if complete else ["maintenance_incomplete"],
        "blockers": list(completeness.get("blockers") or []),
        "storage_validate": dict(storage_validate or {}),
        "run_dir": str(run_dir),
        "stage_plan_path": str(run_dir / "stage_plan.json"),
        "batches_path": str(run_dir / "batches.json"),
        "failed_batches_path": str(run_dir / "failed_batches.json"),
    }
    if early_stop_reason:
        progress["limitations"].append(early_stop_reason)
    _atomic_json_write(run_dir / "progress_summary.json", progress)
    _atomic_json_write(
        run_dir / "failed_batches.json",
        {
            "schema_version": SCHEMA_VERSION,
            "generated_at": progress["generated_at"],
            "run_id": run_id,
            "failed_batch_count": len(failed_batches),
            "failed_symbol_count": len(failed_symbols),
            "failed_batches": failed_batches,
            "failed_symbols": failed_symbols,
        },
    )
    return progress


def _execute_batch(
    *,
    downloader: Any,
    run_dir: Path,
    batch: dict[str, Any],
    effective_target_trade_date: str,
) -> None:
    batch_id = str(batch["batch_id"])
    manifest_path = _batch_manifest_path(run_dir, batch_id)
    started_at = _utc_now_iso()
    batch["status"] = "running"
    batch["updated_at"] = started_at
    batch["manifest_path"] = str(manifest_path)
    _atomic_json_write(
        manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "batch_id": batch_id,
            "category": batch.get("category"),
            "status": "running",
            "started_at": started_at,
            "completed_at": "",
            "target_trade_date": effective_target_trade_date,
            "symbols": list(batch.get("symbols") or []),
            "results": [],
            "successful_symbols": [],
            "failed_symbols": [],
        },
    )

    symbols = list(batch.get("symbols") or [])
    category = str(batch.get("category") or "full_a")
    daily_batch = getattr(downloader, "download_daily_batch", None)
    if not callable(daily_batch):
        raise RuntimeError("staged_noncanonical_daily_batch_api_required")
    pointer_before = _canonical_pointer_state(downloader)
    try:
        results = daily_batch(
            symbols,
            category,
            target_trade_date=effective_target_trade_date,
            publish_canonical=False,
        )
    finally:
        pointer_after = _canonical_pointer_state(downloader)
        if pointer_after != pointer_before:
            raise RuntimeError("staged_canonical_pointer_mutation_detected")
    successful_symbols = [
        str(row.get("symbol") or "").strip().upper()
        for row in results
        if str(row.get("status") or "").lower() in SUCCESS_STATUSES and str(row.get("symbol") or "").strip()
    ]
    failed_symbols = [
        str(row.get("symbol") or "").strip().upper()
        for row in results
        if str(row.get("status") or "").lower() not in SUCCESS_STATUSES and str(row.get("symbol") or "").strip()
    ]
    if not failed_symbols:
        status = COMPLETED_BATCH_STATUS
    elif successful_symbols:
        status = "incomplete"
    else:
        status = "failed"
    batch["status"] = status
    batch["completed_symbol_count"] = len(successful_symbols)
    batch["failed_symbol_count"] = len(failed_symbols)
    batch["updated_at"] = _utc_now_iso()
    _atomic_json_write(
        manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "batch_id": batch_id,
            "category": batch.get("category"),
            "status": status,
            "started_at": started_at,
            "completed_at": batch["updated_at"],
            "target_trade_date": effective_target_trade_date,
            "symbols": list(batch.get("symbols") or []),
            "results": list(results),
            "successful_symbols": successful_symbols,
            "failed_symbols": failed_symbols,
        },
    )


def run_staged_maintenance(
    *,
    market: str = "CN",
    categories: list[str] | None = None,
    batch_size: int = 200,
    max_batches_per_run: int = 1,
    min_symbol_success_rate: float = 0.95,
    target_date: str = "auto",
    daily_window: bool = False,
    resume: bool = False,
    fail_on_incomplete: bool = False,
    allowed_stale_symbols: list[str] | None = None,
    years: int = 3,
    max_workers: int = 4,
    data_dir: str | None = None,
    storage_validate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scoped_market = str(market or "").strip().upper()
    if scoped_market != "CN":
        raise ValueError("staged maintenance currently supports CN only")

    settings = get_market_settings("CN")
    resolved_data_dir = data_dir or settings.data_dir
    if not bool(getattr(CNFullMarketDownloader, "STAGED_NONCANONICAL_SAFE", False)):
        progress_path = _write_noncanonical_writer_block(
            data_dir=resolved_data_dir,
            categories=categories,
            target_date=target_date,
            resume=resume,
            storage_validate=storage_validate,
        )
        raise RuntimeError(
            f"{NONCANONICAL_WRITER_BLOCKER}: progress_summary={progress_path}"
        )
    downloader = CNFullMarketDownloader(
        data_dir=resolved_data_dir,
        years=years,
        max_workers=max_workers,
        batch_size=max(1, int(batch_size or 200)),
    )
    components = downloader.load_components()
    target_categories = downloader._resolve_target_categories(components, categories)
    explicit_target = _compact_trade_date(target_date)
    if explicit_target and str(target_date or "").strip().lower() != "auto":
        same_day_probe = {
            "applicable": False,
            "available": True,
            "reason": "explicit_target_date",
            "trade_date": explicit_target,
        }
    else:
        same_day_probe = downloader._probe_strict_same_day_close_availability(
            components=components,
            target_categories=target_categories,
        )
    target_trade_date, effective_target_trade_date, early_stop_reason = _resolve_effective_target(
        downloader=downloader,
        target_date=target_date,
        same_day_probe=same_day_probe,
    )
    if daily_window:
        _apply_daily_window(downloader, effective_target_trade_date)

    root = _maintenance_root(resolved_data_dir)
    run_dir = _latest_resumable_run(
        root=root,
        categories=target_categories,
        effective_target_trade_date=effective_target_trade_date,
    ) if resume else None
    if run_dir is not None:
        run_id = run_dir.name
        batches = _dedupe_batch_rows(_read_batches_json(run_dir / "batches.json"))
        _write_batches_json(run_dir / "batches.json", batches)
    else:
        run_id = _new_run_id(effective_target_trade_date)
        run_dir = root / run_id
        completeness = downloader.build_completeness_report(
            components=components,
            allowed_stale_symbols=allowed_stale_symbols,
            categories=target_categories,
            target_trade_date=effective_target_trade_date,
            early_stop_reason=early_stop_reason,
        )
        blocking_by_category = _collect_blocking_by_category(downloader, completeness, target_categories)
        batches = _dedupe_batch_rows(_build_batches(blocking_by_category, int(batch_size or 200)))
        stage_plan = {
            "schema_version": SCHEMA_VERSION,
            "created_at": _utc_now_iso(),
            "run_id": run_id,
            "market": "CN",
            "categories": list(target_categories),
            "total_symbols": sum(len(symbols) for symbols in blocking_by_category.values()),
            "batch_size": int(batch_size or 200),
            "max_batches_per_run": int(max_batches_per_run),
            "target_trade_date": target_trade_date,
            "effective_target_trade_date": effective_target_trade_date,
            "strict_trade_date": completeness.get("strict_trade_date"),
            "stable_trade_date": completeness.get("stable_trade_date"),
            "min_symbol_success_rate": float(min_symbol_success_rate),
            "daily_window": bool(daily_window),
            "same_day_close_probe": same_day_probe,
            "early_stop_reason": early_stop_reason,
            "initial_completeness": completeness,
        }
        _atomic_json_write(run_dir / "stage_plan.json", stage_plan)
        _write_batches_json(run_dir / "batches.json", batches)
        initial_progress = _write_progress(
            run_dir=run_dir,
            run_id=run_id,
            categories=target_categories,
            batch_size=int(batch_size or 200),
            max_batches_per_run=int(max_batches_per_run),
            target_trade_date=target_trade_date,
            effective_target_trade_date=effective_target_trade_date,
            min_symbol_success_rate=float(min_symbol_success_rate),
            completeness=completeness,
            batches=batches,
            same_day_probe=same_day_probe,
            early_stop_reason=early_stop_reason,
            storage_validate=storage_validate,
        )
        if completeness.get("complete") or not batches:
            if fail_on_incomplete and not completeness.get("complete"):
                raise RuntimeError("A股 staged maintenance 未完整更新到目标交易日，已按要求终止")
            return {
                "status": initial_progress["status"],
                "maintenance_status": initial_progress["maintenance_status"],
                "run_id": run_id,
                "run_dir": str(run_dir),
                "progress_summary": initial_progress,
                "completeness": completeness,
                "categories": target_categories,
                "failed_symbols": initial_progress["failed_symbols"],
            }

    runnable = [row for row in batches if row.get("status") in RUNNABLE_BATCH_STATUSES]
    for batch in runnable[: max(0, int(max_batches_per_run or 0))]:
        _write_batches_json(run_dir / "batches.json", batches)
        _execute_batch(
            downloader=downloader,
            run_dir=run_dir,
            batch=batch,
            effective_target_trade_date=effective_target_trade_date,
        )
        _write_batches_json(run_dir / "batches.json", batches)

    final_completeness = downloader.build_completeness_report(
        components=components,
        allowed_stale_symbols=allowed_stale_symbols,
        categories=target_categories,
        target_trade_date=effective_target_trade_date,
        early_stop_reason=early_stop_reason,
    )
    progress = _write_progress(
        run_dir=run_dir,
        run_id=run_id,
        categories=target_categories,
        batch_size=int(batch_size or 200),
        max_batches_per_run=int(max_batches_per_run),
        target_trade_date=target_trade_date,
        effective_target_trade_date=effective_target_trade_date,
        min_symbol_success_rate=float(min_symbol_success_rate),
        completeness=final_completeness,
        batches=batches,
        same_day_probe=same_day_probe,
        early_stop_reason=early_stop_reason,
        storage_validate=storage_validate,
    )
    _write_batches_json(run_dir / "batches.json", batches)
    if fail_on_incomplete and not progress.get("complete"):
        raise RuntimeError("A股 staged maintenance 未完整更新到目标交易日，已按要求终止")
    return {
        "status": progress["status"],
        "maintenance_status": progress["maintenance_status"],
        "run_id": run_id,
        "run_dir": str(run_dir),
        "progress_summary": progress,
        "completeness": final_completeness,
        "categories": target_categories,
        "failed_symbols": progress["failed_symbols"],
    }


__all__ = ["run_staged_maintenance"]
