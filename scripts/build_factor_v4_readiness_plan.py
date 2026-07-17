#!/usr/bin/env python3
"""Build a research-only Factor v4 readiness report from one explicit JSON file."""

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

from quant_investor.factors.governance_protocol_v4 import (  # noqa: E402
    assess_factor_governance_readiness_v4,
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
    report = assess_factor_governance_readiness_v4(
        list(payload.get("factor_records", []) or []),
        as_of=str(payload.get("as_of") or ""),
        registry_file_sha256=str(payload.get("registry_file_sha256") or ""),
        production_factor_set_sha256=str(payload.get("production_factor_set_sha256") or ""),
        activation_receipt=(
            dict(payload["activation_receipt"])
            if isinstance(payload.get("activation_receipt"), dict)
            else None
        ),
    )
    _write_private_json(Path(args.output_json).resolve(), report)
    print(f"factor_v4_readiness_status={report['status']}")
    print("production_apply_enabled=false")
    return 0 if report["factor_governance_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
