#!/usr/bin/env python3
"""Build and validate an inert Factor v4 WAL/CAS/inverse transaction plan."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_investor.factors.governance_transaction_v4 import (  # noqa: E402
    build_factor_v4_transaction_plan,
    validate_factor_v4_transaction_plan,
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("input JSON must contain an object")
    return value


def _write_private_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = _read_object(Path(args.input_json).resolve())
    plan = build_factor_v4_transaction_plan(
        transaction_id=str(payload.get("transaction_id") or ""),
        as_of=str(payload.get("as_of") or ""),
        cadence=str(payload.get("cadence") or ""),
        production_factor_count=int(payload.get("production_factor_count", 0)),
        expected_registry_file_sha256=str(payload.get("expected_registry_file_sha256") or ""),
        proposed_registry_file_sha256=str(payload.get("proposed_registry_file_sha256") or ""),
        expected_production_factor_set_sha256=str(
            payload.get("expected_production_factor_set_sha256") or ""
        ),
        proposed_production_factor_set_sha256=str(
            payload.get("proposed_production_factor_set_sha256") or ""
        ),
        proposals=list(payload.get("proposals", []) or []),
        wal_path=str(payload.get("wal_path") or ""),
        inverse_rollback_path=str(payload.get("inverse_rollback_path") or ""),
    )
    validate_factor_v4_transaction_plan(plan)
    _write_private_json(Path(args.output_json).resolve(), plan)
    print(f"factor_v4_transaction_status={plan['status']}")
    print("registry_mutation_performed=false")
    return 0 if plan["status"] == "plan_ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
