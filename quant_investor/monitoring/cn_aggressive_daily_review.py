"""
A股激进科技制造策略日度正式复盘编排入口。

该模块只负责 automation 层的顺序编排：先尝试正式 CN 数据维护，
再运行组合正式复盘。维护失败或不完整时不阻断复盘。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from quant_investor.market.download import run_market_maintenance
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


def _run_maintenance_preflight(args: argparse.Namespace) -> dict[str, Any]:
    if bool(getattr(args, "skip_maintenance", False)):
        return {
            "attempted": False,
            "status": "skipped",
            "non_blocking": True,
            "error": "",
            "elapsed_sec": 0.0,
            "completeness": {},
        }

    started = time.time()
    try:
        result = run_market_maintenance(
            market="CN",
            categories=getattr(args, "categories", None),
            years=int(getattr(args, "maintenance_years", getattr(args, "years", 3))),
            max_workers=int(getattr(args, "maintenance_workers", 4)),
            batch_size=int(getattr(args, "maintenance_batch_size", 50)),
            max_rounds=int(getattr(args, "maintenance_max_rounds", 1)),
            fail_on_incomplete=False,
            allowed_stale_symbols=list(getattr(args, "allowed_stale_symbols", []) or []),
        )
        completeness = dict((result or {}).get("completeness", {}) or {})
        status = "complete" if completeness.get("complete") else "incomplete"
        return {
            "attempted": True,
            "status": status,
            "non_blocking": True,
            "error": "",
            "elapsed_sec": round(time.time() - started, 2),
            "categories": list((result or {}).get("categories", []) or []),
            "completeness": _jsonable(completeness),
        }
    except Exception as exc:
        return {
            "attempted": True,
            "status": "failed_non_blocking",
            "non_blocking": True,
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
    parser.add_argument("--category", action="append", dest="categories")
    parser.add_argument("--maintenance-years", type=int, default=3)
    parser.add_argument("--maintenance-workers", type=int, default=4)
    parser.add_argument("--maintenance-batch-size", type=int, default=50)
    parser.add_argument("--maintenance-max-rounds", type=int, default=1)
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
