#!/usr/bin/env python3
"""Create a Dashboard-only SHA registry for verified archived performance."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

from cn_dashboard_common import (
    DashboardInputError,
    build_history_integrity_registry,
    canonical_json_bytes,
    load_dashboard_catalog_projection,
    load_history_integrity_registry,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORD_ROOT = (
    PROJECT_ROOT
    / "results"
    / "strategy_records"
    / "CN"
    / "aggressive_tech_manufacturing"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "portfolio_dashboard"
    / "private"
    / "generated"
    / "cn_aggressive_history_integrity.v1.json"
)


def publish_registry(
    payload: dict,
    output: Path,
    project_root: Path,
    *,
    integrity_context: dict | None = None,
) -> None:
    rendered = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    previous = output.read_bytes() if output.exists() else None
    with tempfile.TemporaryDirectory(
        prefix="cn-dashboard-history-integrity-"
    ) as temp_name:
        staged = Path(temp_name) / output.name
        staged.write_bytes(rendered)
        with staged.open("rb") as handle:
            os.fsync(handle.fileno())
        try:
            os.replace(staged, output)
            context = integrity_context or {}
            load_history_integrity_registry(
                output,
                project_root,
                intended_generation_id=context.get(
                    "intended_generation_id"
                ),
                dashboard_projection_sha256=context.get(
                    "dashboard_projection_sha256"
                ),
                archive_bindings=context.get("archive_bindings"),
            )
        except Exception:
            if previous is None:
                output.unlink(missing_ok=True)
            else:
                output.write_bytes(previous)
            raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--record-root", type=Path, default=DEFAULT_RECORD_ROOT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--generated-at", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    record_root = args.record_root.resolve()
    projection = load_dashboard_catalog_projection(
        record_root, project_root
    )
    records = projection["historical_records"]
    rejected = projection["historical_rejected"]
    integrity_context = projection.get("integrity_context", {})
    registry_ref = integrity_context.get("history_registry_ref")
    if registry_ref is not None:
        if not isinstance(registry_ref, dict) or not isinstance(
            registry_ref.get("path"), str
        ):
            raise DashboardInputError(
                "catalog_history_registry_ref_invalid"
            )
        bound_output = (project_root / registry_ref["path"]).resolve()
        legacy_default = (
            project_root
            / "portfolio_dashboard/private/generated/"
            "cn_aggressive_history_integrity.v1.json"
        ).resolve()
        requested_output = args.output.resolve()
        if requested_output not in {legacy_default, bound_output}:
            raise DashboardInputError(
                "catalog_history_registry_path_mismatch"
            )
        registry, artifact = load_history_integrity_registry(
            bound_output,
            project_root,
            intended_generation_id=integrity_context.get(
                "intended_generation_id"
            ),
            dashboard_projection_sha256=integrity_context.get(
                "dashboard_projection_sha256"
            ),
            archive_bindings=integrity_context.get("archive_bindings"),
        )
        print(
            canonical_json_bytes(
                {
                    "published": False,
                    "reused_catalog_binding": True,
                    "record_count": len(registry),
                    "content_sha256": integrity_context[
                        "history_registry"
                    ]["content_sha256"],
                    "artifact_sha256": artifact.sha256,
                    "output": str(bound_output),
                }
            ).decode("utf-8")
        )
        return 0
    generated_at = args.generated_at or datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    payload = build_history_integrity_registry(
        records,
        generated_at=generated_at,
        intended_generation_id=integrity_context.get(
            "intended_generation_id"
        ),
        dashboard_projection_sha256=integrity_context.get(
            "dashboard_projection_sha256"
        ),
        archive_bindings=integrity_context.get("archive_bindings"),
    )
    publish_registry(
        payload,
        args.output.resolve(),
        project_root,
        integrity_context=integrity_context,
    )
    print(
        canonical_json_bytes(
            {
                "published": True,
                "record_count": len(records),
                "rejected_record_count": len(rejected),
                "content_sha256": payload["content_sha256"],
                "output": str(args.output.resolve()),
            }
        ).decode("utf-8")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
