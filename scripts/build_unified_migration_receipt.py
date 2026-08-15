#!/usr/bin/env python3
"""Build an idempotent pre-CAS migration receipt; never perform the CAS."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from quant_investor.contracts import canonical_json_bytes
from quant_investor.migration import (
    RULES_RELATIVE_PATH,
    UnifiedCutoverError,
    build_pre_cas_migration_receipt,
    write_pre_cas_migration_receipt,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--rules-path", default=RULES_RELATIVE_PATH)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--archive-plan", type=Path, required=True)
    parser.add_argument("--target-active-pointer", type=Path, required=True)
    parser.add_argument("--target-generation-manifest", type=Path, required=True)
    parser.add_argument("--cutover-id", required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = build_pre_cas_migration_receipt(
            args.workspace_root,
            args.inventory.read_bytes(),
            args.archive_plan.read_bytes(),
            args.target_active_pointer.read_bytes(),
            args.target_generation_manifest.read_bytes(),
            cutover_id=args.cutover_id,
            created_at=args.created_at,
            rules_path=args.rules_path,
        )
        created = write_pre_cas_migration_receipt(args.output, receipt)
    except (OSError, UnifiedCutoverError) as exc:
        code = exc.code if isinstance(exc, UnifiedCutoverError) else "INPUT_UNAVAILABLE"
        sys.stdout.buffer.write(canonical_json_bytes({"code": code, "status": "BLOCKED"}) + b"\n")
        return 2
    sys.stdout.buffer.write(
        canonical_json_bytes(
            {
                "artifact_id": receipt["artifact_id"],
                "cas_performed": False,
                "created": created,
                "output": str(args.output),
                "status": "READY_FOR_CAS",
                "write_performed": False,
            }
        )
        + b"\n"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
