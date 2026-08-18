#!/usr/bin/env python3
"""Atomically publish the read-only CN aggressive Dashboard bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import uuid
from datetime import date, datetime
from pathlib import Path

from quant_investor.strategy_records.store import (
    CATALOG_SCHEMA_V3,
    StrategyRecordStoreError,
    load_registered_catalog,
)

from cn_dashboard_common import (
    DashboardInputError,
    build_bundle,
    validate_bundle_shape,
    verify_source_refs,
)
from cn_dashboard_v2 import (
    DashboardV2Error,
    build_v2_bundle,
    validate_v2_shape,
    verify_v2_source_refs,
)
from cn_dashboard_v2_selector import build_selector, publish_selector
from cn_dashboard_v2_selector import (
    SELECTOR_JSON_FILENAME,
    SELECTOR_JS_FILENAME,
    expected_private_dashboard_output_path,
    require_exact_private_dashboard_output_path,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORD_ROOT = (
    PROJECT_ROOT / "results" / "strategy_records" / "CN" / "aggressive_tech_manufacturing"
)
DEFAULT_BENCHMARK = PROJECT_ROOT / "portfolio_dashboard" / "inputs" / "cn_index_benchmark.csv"
DEFAULT_RISK_FREE = PROJECT_ROOT / "portfolio_dashboard" / "inputs" / "cn_govt_bond_yield.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "portfolio_dashboard" / "private" / "generated"
DEFAULT_JSON = DEFAULT_OUTPUT_DIR / "cn_aggressive_dashboard.v1.json"
DEFAULT_JS = DEFAULT_OUTPUT_DIR / "cn_aggressive_dashboard.v1.js"
DEFAULT_V2_JSON = DEFAULT_OUTPUT_DIR / "cn_aggressive_dashboard.v2.json"
DEFAULT_V2_JS = DEFAULT_OUTPUT_DIR / "cn_aggressive_dashboard.v2.js"
DEFAULT_SELECTOR_JSON = DEFAULT_OUTPUT_DIR / "cn_aggressive_dashboard_selector.v2.json"
DEFAULT_SELECTOR_JS = DEFAULT_OUTPUT_DIR / "cn_aggressive_dashboard_selector.v2.js"
DEFAULT_HISTORY_INTEGRITY = DEFAULT_OUTPUT_DIR / "cn_aggressive_history_integrity.v1.json"

_OUTPUT_FILENAMES = (
    "cn_aggressive_dashboard.v1.json",
    "cn_aggressive_dashboard.v1.js",
    "cn_aggressive_dashboard.v2.json",
    "cn_aggressive_dashboard.v2.js",
    SELECTOR_JSON_FILENAME,
    SELECTOR_JS_FILENAME,
)


def _expected_output_paths(
    project_root: Path,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    return tuple(
        expected_private_dashboard_output_path(project_root, filename)
        for filename in _OUTPUT_FILENAMES
    )  # type: ignore[return-value]


def _require_output_path(project_root: Path, path: Path, filename: str) -> Path:
    try:
        return require_exact_private_dashboard_output_path(
            project_root=project_root,
            path=path,
            filename=filename,
        )
    except ValueError as exc:
        raise DashboardInputError(str(exc)) from exc


def _catalog_history_integrity_path(
    *,
    project_root: Path,
    record_root: Path,
    requested: Path,
) -> Path | None:
    try:
        registered = load_registered_catalog(record_root)
    except StrategyRecordStoreError as exc:
        raise DashboardInputError("record_catalog_invalid:" + str(exc)) from exc
    if registered is None:
        return requested.resolve()
    catalog = registered[1]
    requested_path = requested.resolve()
    legacy_default = (
        project_root / "portfolio_dashboard/private/generated/"
        "cn_aggressive_history_integrity.v1.json"
    ).resolve()
    if catalog.get("schema_id") == CATALOG_SCHEMA_V3:
        default_paths = {legacy_default, DEFAULT_HISTORY_INTEGRITY.resolve()}
        if requested_path not in default_paths:
            raise DashboardInputError("catalog_v3_history_integrity_path_forbidden")
        return None
    ref = catalog.get("history_registry_ref")
    if ref is None:
        return requested_path
    if not isinstance(ref, dict) or not isinstance(ref.get("path"), str):
        raise DashboardInputError("catalog_history_registry_ref_invalid")
    bound_path = (project_root / ref["path"]).resolve()
    if requested_path == legacy_default:
        return bound_path
    return requested_path


def _render_json(bundle: dict) -> bytes:
    return (json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _render_js(bundle: dict) -> bytes:
    payload = json.dumps(bundle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return ("window.MyQuantCNAggressiveDashboard = " + payload + ";\n").encode("utf-8")


def _render_v2_js(bundle: dict) -> bytes:
    payload = json.dumps(bundle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return ("window.MyQuantCNAggressiveDashboardV2 = " + payload + ";\n").encode("utf-8")


def _write_bytes(path: Path, data: bytes) -> None:
    path.write_bytes(data)
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _restore(path: Path, previous: bytes | None) -> None:
    if previous is None:
        if path.exists():
            path.unlink()
        return
    path.write_bytes(previous)


def publish_bundle(bundle: dict, json_path: Path, js_path: Path, project_root: Path) -> None:
    json_path = _require_output_path(project_root, json_path, "cn_aggressive_dashboard.v1.json")
    js_path = _require_output_path(project_root, js_path, "cn_aggressive_dashboard.v1.js")
    errors = validate_bundle_shape(bundle) + verify_source_refs(bundle, project_root)
    if errors:
        raise DashboardInputError("bundle_pre_publish_check_failed:" + ";".join(errors))
    json_bytes = _render_json(bundle)
    js_bytes = _render_js(bundle)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    js_path.parent.mkdir(parents=True, exist_ok=True)
    previous_json = json_path.read_bytes() if json_path.exists() else None
    previous_js = js_path.read_bytes() if js_path.exists() else None
    with tempfile.TemporaryDirectory(prefix="cn-dashboard-stage-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        staged_json = temp_dir / json_path.name
        staged_js = temp_dir / js_path.name
        _write_bytes(staged_json, json_bytes)
        _write_bytes(staged_js, js_bytes)
        staged_readback = json.loads(staged_json.read_text(encoding="utf-8"))
        staged_errors = validate_bundle_shape(staged_readback) + verify_source_refs(
            staged_readback, project_root
        )
        if staged_errors or staged_js.read_bytes() != js_bytes:
            raise DashboardInputError(
                "bundle_staging_check_failed:" + ";".join(staged_errors or ["js_readback_mismatch"])
            )
        try:
            os.replace(staged_json, json_path)
            os.replace(staged_js, js_path)
            published = json.loads(json_path.read_text(encoding="utf-8"))
            published_errors = validate_bundle_shape(published) + verify_source_refs(
                published, project_root
            )
            if published_errors or js_path.read_bytes() != js_bytes:
                raise DashboardInputError(
                    "bundle_post_publish_check_failed:"
                    + ";".join(published_errors or ["js_readback_mismatch"])
                )
        except Exception:
            _restore(json_path, previous_json)
            _restore(js_path, previous_js)
            raise


def publish_bundle_pair(
    *,
    v1_bundle: dict,
    v2_bundle: dict,
    v1_json_path: Path,
    v1_js_path: Path,
    v2_json_path: Path,
    v2_js_path: Path,
    project_root: Path,
) -> None:
    """Stage, validate, and transactionally replace both bundle versions."""

    v1_json_path = _require_output_path(
        project_root, v1_json_path, "cn_aggressive_dashboard.v1.json"
    )
    v1_js_path = _require_output_path(project_root, v1_js_path, "cn_aggressive_dashboard.v1.js")
    v2_json_path = _require_output_path(
        project_root, v2_json_path, "cn_aggressive_dashboard.v2.json"
    )
    v2_js_path = _require_output_path(project_root, v2_js_path, "cn_aggressive_dashboard.v2.js")
    v1_json_bytes = _render_json(v1_bundle)
    errors = (
        validate_bundle_shape(v1_bundle)
        + verify_source_refs(v1_bundle, project_root)
        + validate_v2_shape(v2_bundle)
        + verify_v2_source_refs(
            v2_bundle,
            project_root,
            v1_bytes_override=v1_json_bytes,
        )
    )
    if errors:
        raise DashboardInputError("bundle_pair_pre_publish_check_failed:" + ";".join(errors))
    payloads = {
        v1_json_path: v1_json_bytes,
        v1_js_path: _render_js(v1_bundle),
        v2_json_path: _render_json(v2_bundle),
        v2_js_path: _render_v2_js(v2_bundle),
    }
    for path in payloads:
        path.parent.mkdir(parents=True, exist_ok=True)
    previous = {path: path.read_bytes() if path.exists() else None for path in payloads}
    with tempfile.TemporaryDirectory(prefix="cn-dashboard-v2-stage-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        staged: dict[Path, Path] = {}
        for index, (final_path, raw) in enumerate(payloads.items()):
            stage_path = temp_dir / f"{index}-{final_path.name}"
            _write_bytes(stage_path, raw)
            if stage_path.read_bytes() != raw:
                raise DashboardInputError("bundle_pair_stage_readback_mismatch")
            staged[final_path] = stage_path
        staged_v1 = json.loads(staged[v1_json_path].read_text(encoding="utf-8"))
        staged_v2 = json.loads(staged[v2_json_path].read_text(encoding="utf-8"))
        staged_errors = (
            validate_bundle_shape(staged_v1)
            + verify_source_refs(staged_v1, project_root)
            + validate_v2_shape(staged_v2)
            + verify_v2_source_refs(
                staged_v2,
                project_root,
                v1_bytes_override=staged[v1_json_path].read_bytes(),
            )
        )
        if staged_errors:
            raise DashboardInputError("bundle_pair_staging_check_failed:" + ";".join(staged_errors))
        try:
            for final_path in payloads:
                os.replace(staged[final_path], final_path)
            final_v1 = json.loads(v1_json_path.read_text(encoding="utf-8"))
            final_v2 = json.loads(v2_json_path.read_text(encoding="utf-8"))
            final_errors = (
                validate_bundle_shape(final_v1)
                + verify_source_refs(final_v1, project_root)
                + validate_v2_shape(final_v2)
                + verify_v2_source_refs(final_v2, project_root)
            )
            if (
                final_errors
                or final_v1 != v1_bundle
                or final_v2 != v2_bundle
                or v1_js_path.read_bytes() != payloads[v1_js_path]
                or v2_js_path.read_bytes() != payloads[v2_js_path]
            ):
                raise DashboardInputError(
                    "bundle_pair_post_publish_check_failed:"
                    + ";".join(final_errors or ["pair_readback_mismatch"])
                )
        except Exception:
            for path in reversed(list(payloads)):
                _restore(path, previous[path])
            raise


def _resolve_output_paths(
    args: argparse.Namespace,
    project_root: Path,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    try:
        expected = _expected_output_paths(project_root)
    except ValueError as exc:
        raise DashboardInputError(str(exc)) from exc
    requested = (
        args.json_output,
        args.js_output,
        args.v2_json_output,
        args.v2_js_output,
        args.selector_json_output,
        args.selector_js_output,
    )
    if all(value is None for value in requested):
        return (
            _require_output_path(project_root, expected[0], _OUTPUT_FILENAMES[0]),
            _require_output_path(project_root, expected[1], _OUTPUT_FILENAMES[1]),
            _require_output_path(project_root, expected[2], _OUTPUT_FILENAMES[2]),
            _require_output_path(project_root, expected[3], _OUTPUT_FILENAMES[3]),
            _require_output_path(project_root, expected[4], _OUTPUT_FILENAMES[4]),
            _require_output_path(project_root, expected[5], _OUTPUT_FILENAMES[5]),
        )
    if any(value is None for value in requested):
        raise DashboardInputError("dashboard_output_paths_must_be_complete")
    resolved = tuple(
        _require_output_path(project_root, Path(path), filename)
        for path, filename in zip(requested, _OUTPUT_FILENAMES, strict=True)
    )
    if resolved != expected:
        raise DashboardInputError("dashboard_output_path_forbidden")
    return (
        resolved[0],
        resolved[1],
        resolved[2],
        resolved[3],
        resolved[4],
        resolved[5],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--risk-free", type=Path, default=DEFAULT_RISK_FREE)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--js-output", type=Path)
    parser.add_argument("--v2-json-output", type=Path)
    parser.add_argument("--v2-js-output", type=Path)
    parser.add_argument("--selector-json-output", type=Path)
    parser.add_argument("--selector-js-output", type=Path)
    parser.add_argument("--attempt-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--history-integrity",
        type=Path,
        default=DEFAULT_HISTORY_INTEGRITY,
    )
    parser.add_argument("--generated-at", default=None)
    parser.add_argument("--today", default=None, help="Testing override in YYYY-MM-DD format")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    generated_at = args.generated_at or datetime.now().astimezone().isoformat(timespec="seconds")
    today = date.fromisoformat(args.today) if args.today else date.today()
    selector_paths: tuple[Path, Path] | None = None
    attempt_id = args.attempt_id or (
        "dashboard-v2-"
        + hashlib.sha256((generated_at + "|" + uuid.uuid4().hex).encode("utf-8")).hexdigest()[:24]
    )
    try:
        (
            v1_json_path,
            v1_js_path,
            v2_json_path,
            v2_js_path,
            selector_json_path,
            selector_js_path,
        ) = _resolve_output_paths(args, project_root)
        selector_paths = (selector_json_path, selector_js_path)
        publish_selector(
            build_selector(
                attempt_id=attempt_id,
                status="REFRESHING",
                updated_at=generated_at,
                reason="refresh_started",
            ),
            json_path=selector_json_path,
            js_path=selector_js_path,
            project_root=project_root,
            js_first=True,
        )
        record_root = args.record_root.resolve()
        history_integrity_path = _catalog_history_integrity_path(
            project_root=project_root,
            record_root=record_root,
            requested=args.history_integrity,
        )
        bundle = build_bundle(
            project_root=project_root,
            record_root=record_root,
            benchmark_path=args.benchmark.resolve(),
            risk_free_path=args.risk_free.resolve(),
            generated_at=generated_at,
            today=today,
            history_integrity_path=history_integrity_path,
        )
        if bundle["status"] == "BLOCKED":
            raise DashboardInputError("export_blocked")
        v2_bundle = build_v2_bundle(
            project_root=project_root,
            v1_bundle=bundle,
            v1_json_path=v1_json_path,
            v1_json_bytes_override=_render_json(bundle),
            record_root=record_root,
            generation_local_date=today,
            generated_at=generated_at,
            publication_attempt_id=attempt_id,
        )
        publish_bundle_pair(
            v1_bundle=bundle,
            v2_bundle=v2_bundle,
            v1_json_path=v1_json_path,
            v1_js_path=v1_js_path,
            v2_json_path=v2_json_path,
            v2_js_path=v2_js_path,
            project_root=project_root,
        )
        publish_selector(
            build_selector(
                attempt_id=attempt_id,
                status="UPDATED",
                updated_at=generated_at,
                reason="refresh_completed",
                v2_content_sha256=v2_bundle["content_sha256"],
            ),
            json_path=selector_json_path,
            js_path=selector_js_path,
            project_root=project_root,
            js_first=False,
        )
    except (DashboardInputError, DashboardV2Error, OSError, ValueError) as exc:
        if selector_paths is not None:
            try:
                publish_selector(
                    build_selector(
                        attempt_id=attempt_id,
                        status="BLOCKED",
                        updated_at=generated_at,
                        reason="refresh_failed:" + str(exc)[:512],
                    ),
                    json_path=selector_paths[0],
                    js_path=selector_paths[1],
                    project_root=project_root,
                    js_first=False,
                )
            except (OSError, ValueError):
                pass
        print(
            json.dumps(
                {"exported": False, "status": "BLOCKED", "blocker": str(exc)},
                ensure_ascii=False,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "exported": True,
                "status": bundle["status"],
                "latest_valid_record": bundle["latest_valid_record"],
                "previous_valid_record": bundle["previous_valid_record"],
                "content_sha256": bundle["content_sha256"],
                "v2_content_sha256": v2_bundle["content_sha256"],
                "freshness_status": v2_bundle["freshness"]["status"],
                "mark_as_of": v2_bundle["freshness"]["mark_as_of"],
                "json_output": str(v1_json_path),
                "js_output": str(v1_js_path),
                "v2_json_output": str(v2_json_path),
                "v2_js_output": str(v2_js_path),
                "selector_json_output": str(selector_json_path),
                "selector_js_output": str(selector_js_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
