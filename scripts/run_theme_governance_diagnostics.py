#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from quant_investor.reporting.theme_governance_renderer import (
    render_theme_governance_markdown,
)
from quant_investor.themes.governance import (
    evaluate_theme_governance,
    load_theme_governance_registry,
    write_theme_governance_artifact,
)
from quant_investor.themes.storage import ThemeSnapshotStore


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        snapshot = _load_snapshot(args)
        theme_rotation = _extract_theme_rotation(snapshot)
        registry = load_theme_governance_registry(
            args.registry_json or os.environ.get("THEME_GOVERNANCE_REGISTRY_PATH")
        )
        history = []
        if not args.snapshot_json:
            history = ThemeSnapshotStore(args.snapshot_dir).load_recent(
                market=args.market,
                universe_key=args.universe_key,
                limit=args.history_limit,
            )
        result = evaluate_theme_governance(
            theme_rotation,
            registry=registry,
            history=history,
        )
        payload = result.to_dict()
        market = str(args.market or payload.get("market") or snapshot.get("market") or "CN")
        universe_key = str(
            args.universe_key
            or payload.get("universe_key")
            or snapshot.get("universe_key")
            or "unknown_universe"
        )
        as_of = str(payload.get("as_of") or snapshot.get("as_of") or "unknown_date")
        json_path = write_theme_governance_artifact(
            payload,
            args.output_dir,
            market=market,
            universe_key=universe_key,
            as_of=as_of,
            run_id=args.run_id or "diagnostics",
        )
        markdown_path = json_path.with_suffix(".md")
        markdown_path.write_text(
            render_theme_governance_markdown(payload, max_rows=args.max_rows),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"theme_governance_diagnostics_error: {exc}", file=sys.stderr)
        return 1

    print(f"theme_governance_json={json_path}")
    print(f"theme_governance_markdown={markdown_path}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate local theme governance sidecar artifacts.")
    parser.add_argument("--market", default="CN")
    parser.add_argument("--universe-key", default="full_a")
    parser.add_argument("--snapshot-dir", default="results/theme_snapshots")
    parser.add_argument("--snapshot-json", default="")
    parser.add_argument("--registry-json", default="")
    parser.add_argument("--output-dir", default="results/theme_governance")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--max-rows", type=int, default=20)
    parser.add_argument("--history-limit", type=int, default=10)
    return parser.parse_args(argv)


def _load_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    if args.snapshot_json:
        with Path(args.snapshot_json).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("snapshot_json_root_not_mapping")
        return payload

    payload = ThemeSnapshotStore(args.snapshot_dir).load_latest(
        market=args.market,
        universe_key=args.universe_key,
    )
    if not isinstance(payload, dict):
        raise FileNotFoundError(
            f"no_theme_snapshot_found: market={args.market} universe_key={args.universe_key}"
        )
    return payload


def _extract_theme_rotation(snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
    theme_rotation = snapshot.get("theme_rotation")
    if isinstance(theme_rotation, Mapping):
        return theme_rotation
    if str(snapshot.get("schema_version") or "") == "theme_rotation.v1":
        return snapshot
    raise ValueError("theme_rotation_missing")


if __name__ == "__main__":
    raise SystemExit(main())
