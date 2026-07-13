"""
A股激进科技制造策略日度正式复盘编排入口。

该模块只负责 automation 层的顺序编排：先尝试正式 CN 数据维护，
再运行组合正式复盘。维护失败或不完整时不阻断复盘。
"""

from __future__ import annotations

import argparse
import csv
import os
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

from quant_investor.config import config
from quant_investor.market.download_cn import CNFullMarketDownloader
from quant_investor.market.market_data_store import run_storage_validate
from quant_investor.market.staged_maintenance import run_staged_maintenance
from quant_investor.monitoring import cn_aggressive_portfolio_tracker as tracker
from quant_investor.monitoring.theme_holding_guard import (
    evaluate_holding_theme_guard,
)
from quant_investor.reporting.theme_shadow_renderer import (
    append_theme_production_overlay_section_once,
    append_theme_shadow_section_once,
)
from quant_investor.themes.shadow import build_theme_production_overlay_diagnostics
from quant_investor.themes.storage import ThemeSnapshotStore


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


def _current_theme_production_overlay() -> dict[str, Any]:
    return build_theme_production_overlay_diagnostics(
        funnel_boost_enabled=bool(getattr(config, "THEME_FUNNEL_BOOST_ENABLED", False)),
        risk_guard_enabled=bool(getattr(config, "THEME_RISK_GUARD_ENABLED", False)),
        portfolio_cap_enabled=bool(getattr(config, "THEME_PORTFOLIO_CAP_ENABLED", False)),
    )


def _env_bool_override(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return None


def _formal_review_theme_overrides() -> tuple[dict[str, bool], dict[str, str]]:
    overrides: dict[str, bool] = {}
    rollback_env: dict[str, str] = {}
    default_on_flags = (
        "THEME_FUNNEL_BOOST_ENABLED",
        "THEME_RISK_GUARD_ENABLED",
        "THEME_PORTFOLIO_CAP_ENABLED",
        "THEME_SNAPSHOT_ENABLED",
    )
    for flag in default_on_flags:
        explicit = _env_bool_override(flag)
        if explicit is None:
            overrides[flag] = True
            continue
        overrides[flag] = bool(explicit)
        if not explicit:
            rollback_env[flag] = "0"
    shadow_explicit = _env_bool_override("THEME_SHADOW_MODE_ENABLED")
    if shadow_explicit is not None:
        overrides["THEME_SHADOW_MODE_ENABLED"] = bool(shadow_explicit)
        if shadow_explicit:
            rollback_env["THEME_SHADOW_MODE_ENABLED"] = "1"
    return overrides, rollback_env


@contextmanager
def _temporary_config_overrides(overrides: Mapping[str, bool]):
    original: dict[str, Any] = {}
    try:
        for name, value in overrides.items():
            original[name] = getattr(config, name)
            setattr(config, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(config, name, value)


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _date_key(value: Any) -> str:
    digits = "".join(char for char in str(value or "") if char.isdigit())
    return digits[:8] if len(digits) >= 8 else str(value or "").strip()


def _record_theme_context(
    manifest_payload: Mapping[str, Any],
    market_snapshot_payload: Mapping[str, Any],
) -> dict[str, str]:
    data_snapshot = _as_mapping(manifest_payload.get("data_snapshot")) or _as_mapping(
        market_snapshot_payload.get("data_snapshot")
    )
    dag_status = _as_mapping(manifest_payload.get("candidate_level_dag_status"))
    dag_pipeline = _as_mapping(dag_status.get("dag_pipeline"))
    return {
        "market": str(
            manifest_payload.get("market")
            or market_snapshot_payload.get("market")
            or "CN"
        ),
        "universe_key": str(
            dag_pipeline.get("universe")
            or manifest_payload.get("universe")
            or market_snapshot_payload.get("universe")
            or "full_a"
        ),
        "as_of": str(
            data_snapshot.get("analysis_trade_date")
            or market_snapshot_payload.get("analysis_trade_date")
            or data_snapshot.get("latest_complete_trade_date")
            or data_snapshot.get("latest_trade_date")
            or ""
        ),
    }


def _theme_name(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(
            value.get("theme_name")
            or value.get("name")
            or value.get("theme_id")
            or value.get("id")
            or ""
        ).strip()
    return str(value or "").strip()


def _theme_snapshot_summary(
    *,
    path: Path,
    payload: Mapping[str, Any],
    enabled: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    theme_rotation = _as_mapping(payload.get("theme_rotation"))
    top_themes = [
        name
        for name in (_theme_name(item) for item in list(theme_rotation.get("top_themes") or [])[:10])
        if name
    ]
    theme_scores = _as_mapping(theme_rotation.get("theme_scores"))
    summary = {
        "enabled": bool(enabled),
        "status": "success",
        "path": str(path),
        "artifact_path": str(path),
        "market": str(payload.get("market") or theme_rotation.get("market") or ""),
        "universe_key": str(payload.get("universe_key") or theme_rotation.get("universe_key") or ""),
        "as_of": str(payload.get("as_of") or theme_rotation.get("as_of") or ""),
        "run_id": str(payload.get("run_id") or ""),
        "snapshot_schema_version": str(payload.get("snapshot_schema_version") or ""),
        "theme_rotation_status": str(theme_rotation.get("status") or ""),
        "theme_count": int(len(theme_scores)),
        "top_themes": top_themes,
        "diagnostic_notes": [],
    }
    return summary, theme_rotation


def _load_theme_snapshot_for_record(
    *,
    market: str,
    universe_key: str,
    as_of: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    enabled = bool(getattr(config, "THEME_SNAPSHOT_ENABLED", False))
    if not enabled:
        return (
            {
                "enabled": False,
                "status": "disabled",
                "path": "",
                "artifact_path": "",
                "diagnostic_notes": ["theme_snapshot_disabled"],
            },
            {},
            {},
        )
    try:
        store = ThemeSnapshotStore(
            getattr(config, "THEME_SNAPSHOT_DIR", "results/theme_snapshots")
        )
        latest = store.load_latest_with_path(
            market=market or "CN",
            universe_key=universe_key or "full_a",
            as_of=as_of or None,
        )
    except Exception as exc:
        return (
            {
                "enabled": True,
                "status": "error",
                "path": "",
                "artifact_path": "",
                "error": str(exc),
                "diagnostic_notes": [f"theme_snapshot_lookup_error: {exc}"],
            },
            {},
            {},
        )
    if latest is None:
        note = "theme_snapshot_not_found"
        if as_of:
            note = f"theme_snapshot_not_found_for_as_of:{_date_key(as_of)}"
        return (
            {
                "enabled": True,
                "status": "not_persisted_as_standalone_file",
                "path": "",
                "artifact_path": "",
                "market": market or "CN",
                "universe_key": universe_key or "full_a",
                "as_of": as_of or "",
                "diagnostic_notes": [note],
            },
            {},
            {},
        )
    path, payload = latest
    summary, theme_rotation = _theme_snapshot_summary(
        path=path,
        payload=payload,
        enabled=enabled,
    )
    return summary, theme_rotation, dict(payload)


def _load_theme_shadow_monitor_for_record(
    *,
    market: str,
    universe_key: str,
    as_of: str,
) -> dict[str, Any]:
    enabled = bool(getattr(config, "THEME_SHADOW_MODE_ENABLED", False))
    if not enabled:
        return {
            "status": "disabled",
            "artifact_path": "",
            "final_decision_source": "baseline",
            "diagnostic_notes": ["theme_shadow_disabled"],
        }
    artifact_root = Path(getattr(config, "THEME_SHADOW_ARTIFACT_DIR", "results/theme_shadow"))
    market_dir = artifact_root / (market or "CN")
    candidates: list[Path] = []
    as_of_key = _date_key(as_of)
    if as_of_key:
        exact = market_dir / f"{as_of_key}_{universe_key or 'full_a'}_theme_shadow.json"
        candidates.append(exact)
    if market_dir.exists():
        candidates.extend(
            sorted(
                path
                for path in market_dir.glob("*_theme_shadow.json")
                if path.is_file()
                and (
                    not universe_key
                    or path.name.endswith(f"_{universe_key}_theme_shadow.json")
                )
            )
        )
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if not path.exists():
            continue
        payload = _read_json_mapping(path)
        if not payload:
            continue
        if as_of_key:
            artifact_as_of = _date_key(payload.get("as_of") or path.name.split("_", 1)[0])
            if artifact_as_of and artifact_as_of != as_of_key:
                continue
        monitor = dict(payload)
        monitor["artifact_path"] = str(path)
        monitor.setdefault("status", "success")
        monitor.setdefault("final_decision_source", "baseline")
        return monitor
    note = "theme_shadow_artifact_not_found"
    if as_of:
        note = f"theme_shadow_artifact_not_found_for_as_of:{as_of_key}"
    return {
        "status": "not_persisted",
        "artifact_path": "",
        "market": market or "CN",
        "universe_key": universe_key or "full_a",
        "as_of": as_of or "",
        "final_decision_source": "baseline",
        "diagnostic_notes": [note],
    }


def _write_record_json_artifacts(
    *,
    run_path: Path,
    manifest_payload: dict[str, Any],
    theme_snapshot: Mapping[str, Any],
    snapshot_payload: Mapping[str, Any],
    theme_shadow_monitor: Mapping[str, Any],
) -> None:
    files = manifest_payload.setdefault("files", {})
    if not isinstance(files, dict):
        files = {}
        manifest_payload["files"] = files
    raw_exports = manifest_payload.setdefault("raw_exports", {})
    if not isinstance(raw_exports, dict):
        raw_exports = {}
        manifest_payload["raw_exports"] = raw_exports
    raw_dir = run_path / "raw_exports"
    raw_dir.mkdir(parents=True, exist_ok=True)
    timestamp = str(manifest_payload.get("timestamp") or run_path.name)

    if snapshot_payload:
        record_payload = {
            "theme_snapshot": _jsonable(theme_snapshot),
            "stored_snapshot": _jsonable(snapshot_payload),
        }
        snapshot_path = run_path / "theme_snapshot.json"
        snapshot_path.write_text(
            json.dumps(record_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        files["theme_snapshot"] = "theme_snapshot.json"
        raw_name = f"aggressive_portfolio_{timestamp}_formal_theme_snapshot.json"
        (raw_dir / raw_name).write_text(
            json.dumps(record_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raw_exports["theme_snapshot"] = f"raw_exports/{raw_name}"

    if theme_shadow_monitor and str(theme_shadow_monitor.get("status") or "") != "disabled":
        shadow_path = run_path / "theme_shadow_monitor.json"
        shadow_path.write_text(
            json.dumps(_jsonable(theme_shadow_monitor), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        files["theme_shadow_monitor"] = "theme_shadow_monitor.json"
        raw_name = f"aggressive_portfolio_{timestamp}_formal_theme_shadow_monitor.json"
        (raw_dir / raw_name).write_text(
            json.dumps(_jsonable(theme_shadow_monitor), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raw_exports["theme_shadow_monitor"] = f"raw_exports/{raw_name}"


def _attach_theme_payloads(
    payload: dict[str, Any],
    *,
    overlay: Mapping[str, Any],
    theme_snapshot: Mapping[str, Any],
    theme_rotation: Mapping[str, Any],
    theme_shadow_monitor: Mapping[str, Any],
) -> None:
    payload["theme_production_overlay"] = _jsonable(overlay)
    payload["theme_policy_catalyst"] = _jsonable(overlay.get("theme_policy_catalyst") or {})
    payload["rollback_env"] = _jsonable(overlay.get("rollback_env") or {})
    if theme_snapshot:
        payload["theme_snapshot"] = _jsonable(theme_snapshot)
    if theme_rotation:
        payload["theme_rotation"] = _jsonable(theme_rotation)
    if theme_shadow_monitor:
        payload["theme_shadow_monitor"] = _jsonable(theme_shadow_monitor)

    formal_diagnostics = payload.get("formal_diagnostics")
    if not isinstance(formal_diagnostics, dict):
        formal_diagnostics = {}
        payload["formal_diagnostics"] = formal_diagnostics
    formal_diagnostics["theme_production_overlay"] = _jsonable(overlay)
    if theme_snapshot:
        formal_diagnostics["theme_snapshot"] = _jsonable(theme_snapshot)
    if theme_rotation:
        formal_diagnostics["theme_rotation"] = _jsonable(theme_rotation)
    if theme_shadow_monitor:
        formal_diagnostics["theme_shadow_monitor"] = _jsonable(theme_shadow_monitor)


def _decorate_theme_overlay(
    overlay: Mapping[str, Any],
    *,
    rollback_env: Mapping[str, str],
) -> dict[str, Any]:
    payload = dict(_jsonable(overlay))
    payload["rollback_env"] = dict(rollback_env)
    payload["theme_policy_catalyst"] = {
        "enabled": bool(getattr(config, "THEME_POLICY_CATALYST_ENABLED", False)),
        "status": (
            "enabled"
            if bool(getattr(config, "THEME_POLICY_CATALYST_ENABLED", False))
            else "disabled"
        ),
    }
    diagnostic_notes = list(payload.get("diagnostic_notes") or [])
    if rollback_env:
        diagnostic_notes.append("theme_overlay_explicit_rollback_env")
    payload["diagnostic_notes"] = diagnostic_notes
    return payload


def _append_theme_visibility_sections(
    markdown: str,
    *,
    overlay: Mapping[str, Any],
    theme_shadow_monitor: Mapping[str, Any],
) -> str:
    text = append_theme_production_overlay_section_once(markdown, overlay)
    return append_theme_shadow_section_once(text, theme_shadow_monitor)


def _theme_holding_guard_enabled() -> bool:
    return bool(getattr(config, "THEME_HOLDING_GUARD_ENABLED", False))


def _theme_holding_guard_tighten_ratio() -> float:
    try:
        ratio = float(getattr(config, "THEME_HOLDING_GUARD_TIGHTEN_RATIO", 0.5))
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, ratio)


def _guard_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _read_holding_guard_rows(run_path: Path) -> list[dict[str, str]]:
    path = run_path / "holdings_review.csv"
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except (OSError, UnicodeDecodeError, csv.Error):
        return []


def _holding_guard_display_stop(row: Mapping[str, Any], ratio: float) -> float:
    current_price = _guard_float(row.get("current_price"))
    stage_stop = _guard_float(row.get("stage_stop_price"))
    if current_price <= 0 or stage_stop <= 0:
        return stage_stop
    buffer = max(current_price - stage_stop, 0.0)
    return round(current_price - buffer * ratio, 2)


def _build_theme_holding_guard_payload(
    *,
    rows: list[dict[str, str]],
    theme_snapshot: Mapping[str, Any],
    theme_rotation: Mapping[str, Any],
    snapshot_payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not _theme_holding_guard_enabled():
        return {"enabled": False, "status": "disabled", "signals": []}
    if not rows:
        return {
            "enabled": True,
            "status": "no_holdings_review",
            "signals": [],
            "diagnostic_notes": ["holdings_review_unavailable"],
        }
    if not snapshot_payload and not theme_rotation:
        return {
            "enabled": True,
            "status": "snapshot_unavailable",
            "signals": [],
            "diagnostic_notes": ["theme snapshot unavailable"],
        }

    symbols = [str(row.get("symbol") or "").strip().upper() for row in rows]
    payload = dict(snapshot_payload or {})
    if theme_rotation and "theme_rotation" not in payload:
        payload["theme_rotation"] = dict(theme_rotation)
    signals = evaluate_holding_theme_guard(symbols, payload)
    serialized = [signal.to_dict() for signal in signals.values()]
    return {
        "enabled": True,
        "status": "success",
        "tighten_count": sum(1 for signal in signals.values() if signal.guard_level == "tighten"),
        "watch_count": sum(1 for signal in signals.values() if signal.guard_level == "watch"),
        "signals": serialized,
        "theme_snapshot_status": str(theme_snapshot.get("status") or ""),
        "diagnostic_notes": [],
    }


def _theme_holding_guard_section(
    *,
    rows: list[dict[str, str]],
    payload: Mapping[str, Any],
    ratio: float,
) -> str:
    if not payload.get("enabled"):
        return ""
    lines = ["## 主题状态守卫"]
    status = str(payload.get("status") or "")
    if status == "snapshot_unavailable":
        lines.append("- theme snapshot unavailable")
        return "\n".join(lines) + "\n"
    if status == "no_holdings_review":
        lines.append("- holdings_review unavailable")
        return "\n".join(lines) + "\n"

    rows_by_symbol = {
        str(row.get("symbol") or "").strip().upper(): row
        for row in rows
        if str(row.get("symbol") or "").strip()
    }
    active = [
        signal
        for signal in list(payload.get("signals") or [])
        if isinstance(signal, Mapping) and str(signal.get("guard_level") or "") in {"watch", "tighten"}
    ]
    if not active:
        lines.append("- 当前持仓主题阶段未触发 watch/tighten。")
        return "\n".join(lines) + "\n"

    for signal in active:
        symbol = str(signal.get("symbol") or "").strip().upper()
        row = rows_by_symbol.get(symbol, {})
        name = str(row.get("name") or "").strip()
        phase = str(signal.get("phase") or "")
        guard_level = str(signal.get("guard_level") or "")
        theme_name = str(signal.get("primary_theme_name") or signal.get("primary_theme_id") or "")
        reasons = ", ".join(str(item) for item in list(signal.get("reasons") or []))
        prefix = f"- `{symbol}` {name}".rstrip()
        if guard_level == "tighten":
            stage_stop = _guard_float(row.get("stage_stop_price"))
            display_stop = _holding_guard_display_stop(row, ratio)
            lines.append(
                f"{prefix}: guard=tighten；theme={theme_name}；phase={phase}；"
                f"原止损 {stage_stop:.2f} -> 展示止损 {display_stop:.2f}；"
                f"主题转弱（{phase}），止损缓冲收紧；原因 {reasons}"
            )
        else:
            lines.append(
                f"{prefix}: guard=watch；theme={theme_name}；phase={phase}；原因 {reasons}"
            )
    return "\n".join(lines) + "\n"


def _append_theme_holding_guard_section(markdown: str, section: str) -> str:
    if not section or "## 主题状态守卫" in markdown:
        return markdown
    return f"{markdown.rstrip()}\n\n{section}".rstrip() + "\n"


def _attach_theme_holding_guard_payload(
    run_path: Path,
    payload: Mapping[str, Any],
) -> None:
    for name in ("manifest.json", "market_snapshot.json"):
        path = run_path / name
        if not path.exists():
            continue
        record = _read_json_mapping(path)
        if not record:
            continue
        record["theme_holding_guard"] = _jsonable(payload)
        diagnostics = record.get("formal_diagnostics")
        if not isinstance(diagnostics, dict):
            diagnostics = {}
            record["formal_diagnostics"] = diagnostics
        diagnostics["theme_holding_guard"] = _jsonable(payload)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def _apply_theme_holding_guard_to_record(
    run_dir: str | Path,
    *,
    theme_snapshot: Mapping[str, Any],
    theme_rotation: Mapping[str, Any],
    snapshot_payload: Mapping[str, Any],
) -> None:
    if not _theme_holding_guard_enabled():
        return
    run_path = Path(run_dir)
    rows = _read_holding_guard_rows(run_path)
    payload = _build_theme_holding_guard_payload(
        rows=rows,
        theme_snapshot=theme_snapshot,
        theme_rotation=theme_rotation,
        snapshot_payload=snapshot_payload,
    )
    _attach_theme_holding_guard_payload(run_path, payload)
    section = _theme_holding_guard_section(
        rows=rows,
        payload=payload,
        ratio=_theme_holding_guard_tighten_ratio(),
    )

    report_paths = [run_path / "analysis_report.md"]
    manifest_payload = _read_json_mapping(run_path / "manifest.json")
    raw_exports = manifest_payload.get("raw_exports") if isinstance(manifest_payload, dict) else {}
    if isinstance(raw_exports, dict) and raw_exports.get("report"):
        report_paths.append(run_path / str(raw_exports["report"]))
    notes_path = run_path.parent / "latest_notes_payload.md"
    if notes_path.exists():
        report_paths.append(notes_path)
    for report_path in report_paths:
        if not report_path.exists():
            continue
        text = report_path.read_text(encoding="utf-8")
        updated = _append_theme_holding_guard_section(text, section)
        if updated != text:
            report_path.write_text(updated, encoding="utf-8")


def _attach_theme_overlay_to_record(
    run_dir: str | Path,
    overlay: dict[str, Any],
) -> None:
    run_path = Path(run_dir)
    if not run_path.exists():
        return
    payloads: dict[str, dict[str, Any]] = {}
    for name in ("manifest.json", "market_snapshot.json"):
        path = run_path / name
        if not path.exists():
            continue
        payloads[name] = _read_json_mapping(path)
    manifest_payload = payloads.get("manifest.json", {})
    market_snapshot_payload = payloads.get("market_snapshot.json", {})
    context = _record_theme_context(manifest_payload, market_snapshot_payload)
    theme_snapshot, theme_rotation, snapshot_payload = _load_theme_snapshot_for_record(
        market=context["market"],
        universe_key=context["universe_key"],
        as_of=context["as_of"],
    )
    theme_shadow_monitor = _load_theme_shadow_monitor_for_record(
        market=context["market"],
        universe_key=context["universe_key"],
        as_of=context["as_of"],
    )
    if theme_snapshot and theme_shadow_monitor:
        snapshot_path = str(theme_snapshot.get("path") or theme_snapshot.get("artifact_path") or "")
        if snapshot_path:
            theme_shadow_monitor = dict(theme_shadow_monitor)
            theme_shadow_monitor.setdefault("theme_snapshot_path", snapshot_path)
            theme_shadow_monitor.setdefault("theme_snapshot", _jsonable(theme_snapshot))

    if manifest_payload:
        _write_record_json_artifacts(
            run_path=run_path,
            manifest_payload=manifest_payload,
            theme_snapshot=theme_snapshot,
            snapshot_payload=snapshot_payload,
            theme_shadow_monitor=theme_shadow_monitor,
        )

    for name, payload in payloads.items():
        _attach_theme_payloads(
            payload,
            overlay=overlay,
            theme_snapshot=theme_snapshot,
            theme_rotation=theme_rotation,
            theme_shadow_monitor=theme_shadow_monitor,
        )
        path = run_path / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report_paths = [run_path / "analysis_report.md"]
    raw_exports = manifest_payload.get("raw_exports") if isinstance(manifest_payload, dict) else {}
    if isinstance(raw_exports, dict) and raw_exports.get("report"):
        report_paths.append(run_path / str(raw_exports["report"]))
    notes_path = run_path.parent / "latest_notes_payload.md"
    if notes_path.exists():
        report_paths.append(notes_path)
    for report_path in report_paths:
        if not report_path.exists():
            continue
        report_text = report_path.read_text(encoding="utf-8")
        updated = _append_theme_visibility_sections(
            report_text,
            overlay=overlay,
            theme_shadow_monitor=theme_shadow_monitor,
        )
        if updated != report_text:
            report_path.write_text(updated, encoding="utf-8")
    _apply_theme_holding_guard_to_record(
        run_path,
        theme_snapshot=theme_snapshot,
        theme_rotation=theme_rotation,
        snapshot_payload=snapshot_payload,
    )


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
    theme_overrides, rollback_env = _formal_review_theme_overrides()
    with _temporary_config_overrides(theme_overrides):
        result = tracker.run_tracker(tracker_args)
        result["maintenance_preflight"] = preflight
        result["full_market_metrics_cache"] = _jsonable(
            result.get("full_market_metrics_cache")
            or result.get("market_metrics_prewarm")
            or {}
        )
        theme_production_overlay = _decorate_theme_overlay(
            _current_theme_production_overlay(),
            rollback_env=rollback_env,
        )
        result["theme_production_overlay"] = theme_production_overlay
        result["rollback_env"] = dict(rollback_env)
        result["theme_policy_catalyst"] = theme_production_overlay.get("theme_policy_catalyst")
        if result.get("run_dir"):
            _attach_preflight_to_record(result["run_dir"], preflight)
            _attach_theme_overlay_to_record(result["run_dir"], theme_production_overlay)
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
