"""
A股激进科技制造策略日度正式复盘编排入口。

该模块只负责 automation 层的顺序编排：先尝试正式 CN 数据维护，
再运行组合正式复盘。维护失败或不完整时不阻断复盘。
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from quant_investor.market.download_cn import CNFullMarketDownloader
from quant_investor.market.market_data_store import run_storage_validate
from quant_investor.market.staged_maintenance import run_staged_maintenance
from quant_investor.monitoring import cn_aggressive_portfolio_tracker as tracker


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _latest_healthy_snapshot(storage_validate: dict[str, Any]) -> dict[str, Any]:
    if str(storage_validate.get("status") or "").lower() != "passed":
        return {}
    return {
        "snapshot_id": str(storage_validate.get("snapshot_id") or ""),
        "latest_complete_trade_date": str(storage_validate.get("latest_complete_trade_date") or ""),
        "latest_trade_date": str(storage_validate.get("latest_trade_date") or ""),
        "manifest_path": str(storage_validate.get("manifest_path") or ""),
        "latest_pointer_path": str(storage_validate.get("latest_pointer_path") or ""),
        "coverage": _jsonable(storage_validate.get("coverage") or {}),
        "coverage_ratio": storage_validate.get("coverage_ratio"),
    }


def _storage_coverage_ratio(
    storage_validate: dict[str, Any],
    *,
    expected_symbol_count: int | None = None,
) -> float | None:
    coverage = storage_validate.get("coverage") if isinstance(storage_validate.get("coverage"), dict) else {}
    candidates = [
        storage_validate.get("coverage_ratio"),
        coverage.get("coverage_ratio"),
    ]
    for candidate in candidates:
        try:
            if candidate is not None:
                return float(candidate)
        except (TypeError, ValueError):
            continue
    try:
        expected = int(expected_symbol_count or 0)
        covered = int(coverage.get("symbol_count") or coverage.get("covered_count") or 0)
        if expected > 0:
            return min(1.0, max(0.0, covered / expected))
    except (TypeError, ValueError):
        pass
    return None


def _expected_symbol_count(components: dict[str, Any], categories: list[str]) -> int:
    symbols = {
        str(symbol or "").strip().upper()
        for category in categories
        for symbol in components.get(category, []) or []
        if str(symbol or "").strip()
    }
    if symbols:
        return len(symbols)
    stats = components.get("stats") if isinstance(components.get("stats"), dict) else {}
    try:
        return int(stats.get("total_unique") or 0)
    except (TypeError, ValueError):
        return 0


def _build_quick_preflight_probe(args: argparse.Namespace) -> dict[str, Any]:
    downloader = CNFullMarketDownloader(
        years=int(getattr(args, "maintenance_years", getattr(args, "years", 3))),
        max_workers=int(getattr(args, "maintenance_workers", 4)),
        batch_size=int(getattr(args, "maintenance_batch_size", 200)),
    )
    components = downloader.load_components()
    categories = downloader._resolve_target_categories(
        components,
        getattr(args, "categories", None),
    )
    same_day_probe = downloader._probe_strict_same_day_close_availability(
        components=components,
        target_categories=categories,
    )
    explicit_target = str(getattr(args, "target_date", "auto") or "auto").strip()
    return {
        "components": components,
        "categories": categories,
        "expected_symbol_count": _expected_symbol_count(components, categories),
        "same_day_close_probe": same_day_probe,
        "explicit_target": explicit_target,
    }


def _run_maintenance_preflight(args: argparse.Namespace) -> dict[str, Any]:
    if bool(getattr(args, "skip_maintenance", False)):
        return {
            "attempted": False,
            "status": "skipped",
            "maintenance_status": "skipped",
            "non_blocking": True,
            "parquet_canonical_status": "not_checked",
            "decision_data_status": "unknown",
            "latest_healthy_snapshot": {},
            "staged_progress": {},
            "remaining_batches": None,
            "failed_symbols": [],
            "limitations": ["skip_maintenance_requested"],
            "blockers": [],
            "error": "",
            "elapsed_sec": 0.0,
            "completeness": {},
        }

    started = time.time()
    try:
        storage = run_storage_validate(market="CN")
        parquet_healthy = str(storage.get("status") or "").lower() == "passed"
        latest_snapshot = _latest_healthy_snapshot(storage)
        if not parquet_healthy:
            blockers = list(storage.get("blockers") or [])
            return {
                "attempted": True,
                "status": "failed_non_blocking",
                "maintenance_status": "skipped_parquet_unhealthy",
                "non_blocking": True,
                "parquet_canonical_status": "unhealthy",
                "decision_data_status": "unavailable",
                "latest_healthy_snapshot": {},
                "staged_progress": {},
                "remaining_batches": None,
                "failed_symbols": [],
                "limitations": ["strict_parquet_unhealthy"],
                "blockers": blockers,
                "error": "; ".join(str(item) for item in blockers),
                "elapsed_sec": round(time.time() - started, 2),
                "storage_validate": _jsonable(storage),
                "completeness": {},
            }

        probe = _build_quick_preflight_probe(args)
        explicit_target = str(probe.get("explicit_target") or "auto").strip()
        same_day_probe = dict(probe.get("same_day_close_probe", {}) or {})
        if explicit_target.lower() != "auto":
            effective_target = explicit_target
            early_stop_reason = ""
        elif same_day_probe.get("applicable") and same_day_probe.get("available") is True:
            effective_target = str(same_day_probe.get("trade_date") or "")
            early_stop_reason = ""
        else:
            effective_target = str(latest_snapshot.get("latest_complete_trade_date") or "")
            early_stop_reason = (
                "strict_same_day_unavailable"
                if same_day_probe.get("applicable") and same_day_probe.get("available") is False
                else ""
            )
        completeness = tracker.build_parquet_canonical_completeness_report(
            reader=tracker.MarketDataReader(market="CN"),
            components=dict(probe.get("components", {}) or {}),
            categories=list(probe.get("categories", []) or []),
            allowed_stale_symbols=list(getattr(args, "allowed_stale_symbols", []) or []),
            target_trade_date=effective_target,
            early_stop_reason=early_stop_reason,
        )
        coverage_ratio = _storage_coverage_ratio(
            storage,
            expected_symbol_count=int(probe.get("expected_symbol_count") or 0),
        )
        if coverage_ratio is None:
            coverage_ratio = float(completeness.get("coverage_ratio") or 0.0)
        min_success = float(getattr(args, "min_symbol_success_rate", 0.95))
        same_day_unavailable = same_day_probe.get("applicable") and same_day_probe.get("available") is False
        if same_day_unavailable and coverage_ratio >= min_success:
            return {
                "attempted": True,
                "status": "skipped",
                "maintenance_status": "skipped_same_day_unavailable",
                "non_blocking": True,
                "parquet_canonical_status": "healthy",
                "decision_data_status": "sufficient_limited",
                "latest_healthy_snapshot": latest_snapshot,
                "staged_progress": {},
                "remaining_batches": None,
                "failed_symbols": [],
                "limitations": ["strict_same_day_unavailable", "using_latest_healthy_snapshot"],
                "blockers": [],
                "error": "",
                "elapsed_sec": round(time.time() - started, 2),
                "storage_validate": _jsonable(storage),
                "same_day_close_probe": _jsonable(same_day_probe),
                "categories": list(probe.get("categories", []) or []),
                "completeness": _jsonable(completeness),
            }

        if completeness.get("complete"):
            return {
                "attempted": True,
                "status": "complete",
                "maintenance_status": "complete",
                "non_blocking": True,
                "parquet_canonical_status": "healthy",
                "decision_data_status": "sufficient",
                "latest_healthy_snapshot": latest_snapshot,
                "staged_progress": {},
                "remaining_batches": 0,
                "failed_symbols": [],
                "limitations": [],
                "blockers": [],
                "error": "",
                "elapsed_sec": round(time.time() - started, 2),
                "storage_validate": _jsonable(storage),
                "same_day_close_probe": _jsonable(same_day_probe),
                "categories": list(probe.get("categories", []) or []),
                "completeness": _jsonable(completeness),
            }

        staged_result = run_staged_maintenance(
            market="CN",
            categories=getattr(args, "categories", None),
            years=int(getattr(args, "maintenance_years", getattr(args, "years", 3))),
            max_workers=int(getattr(args, "maintenance_workers", 4)),
            batch_size=int(getattr(args, "maintenance_batch_size", 200)),
            max_batches_per_run=int(getattr(args, "maintenance_max_batches_per_run", 200)),
            min_symbol_success_rate=min_success,
            target_date=str(getattr(args, "target_date", "auto") or "auto"),
            daily_window=bool(getattr(args, "daily_window", True)),
            resume=True,
            fail_on_incomplete=False,
            allowed_stale_symbols=list(getattr(args, "allowed_stale_symbols", []) or []),
            storage_validate=storage,
        )
        staged_payload = dict(staged_result or {})
        progress = dict(staged_payload.get("progress_summary", {}) or {})
        decision_status = str(progress.get("decision_data_status") or "limited")
        return {
            "attempted": True,
            "status": str(progress.get("status") or staged_payload.get("status") or "incomplete"),
            "maintenance_status": str(
                progress.get("maintenance_status") or staged_payload.get("maintenance_status") or "incomplete"
            ),
            "non_blocking": True,
            "parquet_canonical_status": "healthy",
            "decision_data_status": decision_status,
            "latest_healthy_snapshot": latest_snapshot,
            "staged_progress": _jsonable(progress),
            "remaining_batches": progress.get("remaining_batches"),
            "failed_symbols": list(progress.get("failed_symbols") or staged_payload.get("failed_symbols") or []),
            "limitations": list(progress.get("limitations") or ["maintenance_incomplete"]),
            "blockers": list(progress.get("blockers") or []),
            "error": "",
            "elapsed_sec": round(time.time() - started, 2),
            "storage_validate": _jsonable(storage),
            "same_day_close_probe": _jsonable(same_day_probe),
            "categories": list(staged_payload.get("categories", []) or probe.get("categories", []) or []),
            "completeness": _jsonable(staged_payload.get("completeness") or completeness),
        }
    except Exception as exc:
        return {
            "attempted": True,
            "status": "failed_non_blocking",
            "maintenance_status": "failed_non_blocking",
            "non_blocking": True,
            "parquet_canonical_status": "unknown",
            "decision_data_status": "unknown",
            "latest_healthy_snapshot": {},
            "staged_progress": {},
            "remaining_batches": None,
            "failed_symbols": [],
            "limitations": ["maintenance_preflight_exception"],
            "blockers": [str(exc)],
            "error": str(exc),
            "elapsed_sec": round(time.time() - started, 2),
            "completeness": {},
        }


def _attach_preflight_to_record(run_dir: str | Path, preflight: dict[str, Any]) -> None:
    run_path = Path(run_dir)
    for name in ("manifest.json", "market_snapshot.json"):
        path = run_path / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["maintenance_preflight"] = _jsonable(preflight)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_daily_review(args: argparse.Namespace) -> dict[str, Any]:
    preflight = _run_maintenance_preflight(args)
    tracker_args = argparse.Namespace(
        base_dir=getattr(args, "base_dir", str(tracker.DEFAULT_BASE_DIR)),
        years=int(getattr(args, "years", 7)),
        max_rounds=int(getattr(args, "tracker_max_rounds", 3)),
        source_record=getattr(args, "source_record", None),
        allowed_stale_symbols=list(getattr(args, "allowed_stale_symbols", []) or []),
        skip_market_metrics_prewarm=bool(getattr(args, "skip_market_metrics_prewarm", False)),
        advisory_only=bool(getattr(args, "advisory_only", True)),
        quote_input_json=str(getattr(args, "quote_input_json", "") or ""),
        allow_live_quotes=bool(getattr(args, "allow_live_quotes", False)),
        quote_max_age_seconds=int(
            getattr(
                args,
                "quote_max_age_seconds",
                tracker.DEFAULT_QUOTE_MAX_AGE_SECONDS,
            )
        ),
        decision_log_path=str(
            getattr(args, "decision_log_path", tracker.DEFAULT_DECISION_LOG_PATH)
            or tracker.DEFAULT_DECISION_LOG_PATH
        ),
    )
    result = tracker.run_tracker(tracker_args)
    result["maintenance_preflight"] = preflight
    result["full_market_metrics_cache"] = _jsonable(
        result.get("full_market_metrics_cache")
        or result.get("market_metrics_prewarm")
        or {}
    )
    if result.get("run_dir"):
        _attach_preflight_to_record(result["run_dir"], preflight)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A股激进科技制造策略维护优先正式复盘编排器")
    parser.add_argument("--base-dir", default=str(tracker.DEFAULT_BASE_DIR))
    parser.add_argument("--years", type=int, default=7)
    parser.add_argument("--tracker-max-rounds", type=int, default=3)
    parser.add_argument("--source-record", default=None)
    parser.add_argument("--allowed-stale-symbols", nargs="*", default=[])
    execution_mode = parser.add_mutually_exclusive_group()
    execution_mode.add_argument(
        "--advisory-only",
        dest="advisory_only",
        action="store_true",
        help="仅写建议和 pending/rejected 记录，不写本地模拟成交（默认）",
    )
    execution_mode.add_argument(
        "--allow-local-manual-fills",
        dest="advisory_only",
        action="store_false",
        help="显式授权在所有门禁通过后写本地/manual paper fill；仍不调用券商",
    )
    parser.set_defaults(advisory_only=True)
    quote_source = parser.add_mutually_exclusive_group()
    quote_source.add_argument("--quote-input-json", default="")
    quote_source.add_argument("--allow-live-quotes", action="store_true", default=False)
    parser.add_argument(
        "--quote-max-age-seconds",
        type=int,
        default=tracker.DEFAULT_QUOTE_MAX_AGE_SECONDS,
    )
    parser.add_argument(
        "--decision-log-path",
        default=str(tracker.DEFAULT_DECISION_LOG_PATH),
    )
    parser.add_argument("--category", action="append", dest="categories")
    parser.add_argument("--maintenance-years", type=int, default=3)
    parser.add_argument("--maintenance-workers", type=int, default=4)
    parser.add_argument("--maintenance-batch-size", type=int, default=200)
    parser.add_argument("--maintenance-max-rounds", type=int, default=1)
    parser.add_argument("--maintenance-max-batches-per-run", type=int, default=200)
    parser.add_argument("--min-symbol-success-rate", type=float, default=0.95)
    parser.add_argument("--target-date", default="auto")
    parser.add_argument("--daily-window", action="store_true", default=True)
    parser.add_argument("--skip-maintenance", action="store_true")
    parser.add_argument(
        "--skip-market-metrics-prewarm",
        action="store_true",
        help="工程排障用：跳过启动前 full-market metrics 缓存预热",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_daily_review(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
