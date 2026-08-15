#!/usr/bin/env python3
"""Resolve one deterministic unified-cutover inventory without runtime writes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from quant_investor.contracts import canonical_json_bytes
from quant_investor.migration import (
    RULES_RELATIVE_PATH,
    UnifiedCutoverError,
    resolve_unified_cutover_inventory,
    write_inventory,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--rules-path", default=RULES_RELATIVE_PATH)
    parser.add_argument("--codex-home", type=Path)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        resolution = resolve_unified_cutover_inventory(
            args.workspace_root,
            created_at=args.created_at,
            rules_path=args.rules_path,
            codex_home=args.codex_home,
        )
        created = write_inventory(args.output, resolution)
    except UnifiedCutoverError as exc:
        sys.stdout.buffer.write(
            canonical_json_bytes({"code": exc.code, "status": "BLOCKED"}) + b"\n"
        )
        return 2
    sys.stdout.buffer.write(
        canonical_json_bytes(
            {
                "artifact_id": resolution.document["artifact_id"],
                "created": created,
                "output": str(args.output),
                "status": "COMPLETE",
            }
        )
        + b"\n"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
