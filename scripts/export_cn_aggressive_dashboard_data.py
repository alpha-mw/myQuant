#!/usr/bin/env python3
"""Atomically publish the read-only CN aggressive Dashboard bundle."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import date, datetime
from pathlib import Path

from cn_dashboard_common import (
    DashboardInputError,
    build_bundle,
    validate_bundle_shape,
    verify_source_refs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORD_ROOT = (
    PROJECT_ROOT
    / "results"
    / "strategy_records"
    / "CN"
    / "aggressive_tech_manufacturing"
)
DEFAULT_BENCHMARK = (
    PROJECT_ROOT / "portfolio_dashboard" / "inputs" / "cn_index_benchmark.csv"
)
DEFAULT_RISK_FREE = (
    PROJECT_ROOT / "portfolio_dashboard" / "inputs" / "cn_govt_bond_yield.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "portfolio_dashboard" / "private" / "generated"
)
DEFAULT_JSON = DEFAULT_OUTPUT_DIR / "cn_aggressive_dashboard.v1.json"
DEFAULT_JS = DEFAULT_OUTPUT_DIR / "cn_aggressive_dashboard.v1.js"
DEFAULT_HISTORY_INTEGRITY = (
    DEFAULT_OUTPUT_DIR / "cn_aggressive_history_integrity.v1.json"
)


def _render_json(bundle: dict) -> bytes:
    return (
        json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _render_js(bundle: dict) -> bytes:
    payload = json.dumps(
        bundle, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return ("window.MyQuantCNAggressiveDashboard = " + payload + ";\n").encode(
        "utf-8"
    )


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


def publish_bundle(
    bundle: dict, json_path: Path, js_path: Path, project_root: Path
) -> None:
    errors = validate_bundle_shape(bundle) + verify_source_refs(
        bundle, project_root
    )
    if errors:
        raise DashboardInputError(
            "bundle_pre_publish_check_failed:" + ";".join(errors)
        )
    json_bytes = _render_json(bundle)
    js_bytes = _render_js(bundle)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    js_path.parent.mkdir(parents=True, exist_ok=True)
    previous_json = json_path.read_bytes() if json_path.exists() else None
    previous_js = js_path.read_bytes() if js_path.exists() else None
    with tempfile.TemporaryDirectory(
        prefix="cn-dashboard-stage-"
    ) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        staged_json = temp_dir / json_path.name
        staged_js = temp_dir / js_path.name
        _write_bytes(staged_json, json_bytes)
        _write_bytes(staged_js, js_bytes)
        staged_readback = json.loads(staged_json.read_text(encoding="utf-8"))
        staged_errors = validate_bundle_shape(
            staged_readback
        ) + verify_source_refs(staged_readback, project_root)
        if staged_errors or staged_js.read_bytes() != js_bytes:
            raise DashboardInputError(
                "bundle_staging_check_failed:"
                + ";".join(staged_errors or ["js_readback_mismatch"])
            )
        try:
            os.replace(staged_json, json_path)
            os.replace(staged_js, js_path)
            published = json.loads(json_path.read_text(encoding="utf-8"))
            published_errors = validate_bundle_shape(
                published
            ) + verify_source_refs(published, project_root)
            if published_errors or js_path.read_bytes() != js_bytes:
                raise DashboardInputError(
                    "bundle_post_publish_check_failed:"
                    + ";".join(published_errors or ["js_readback_mismatch"])
                )
        except Exception:
            _restore(json_path, previous_json)
            _restore(js_path, previous_js)
            raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--record-root", type=Path, default=DEFAULT_RECORD_ROOT
    )
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--risk-free", type=Path, default=DEFAULT_RISK_FREE)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--js-output", type=Path, default=DEFAULT_JS)
    parser.add_argument(
        "--history-integrity",
        type=Path,
        default=DEFAULT_HISTORY_INTEGRITY,
    )
    parser.add_argument("--generated-at", default=None)
    parser.add_argument(
        "--today", default=None, help="Testing override in YYYY-MM-DD format"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generated_at = args.generated_at or datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    today = date.fromisoformat(args.today) if args.today else date.today()
    try:
        bundle = build_bundle(
            project_root=args.project_root.resolve(),
            record_root=args.record_root.resolve(),
            benchmark_path=args.benchmark.resolve(),
            risk_free_path=args.risk_free.resolve(),
            generated_at=generated_at,
            today=today,
            history_integrity_path=args.history_integrity.resolve(),
        )
        if bundle["status"] == "BLOCKED":
            raise DashboardInputError("export_blocked")
        publish_bundle(
            bundle,
            args.json_output.resolve(),
            args.js_output.resolve(),
            args.project_root.resolve(),
        )
    except DashboardInputError as exc:
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
                "json_output": str(args.json_output),
                "js_output": str(args.js_output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
