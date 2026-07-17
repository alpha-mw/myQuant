#!/usr/bin/env python3
"""Validate an explicit Factor v4 replay/evidence file without activation."""

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

from quant_investor.factors.governance_canonical_replay_v4 import (  # noqa: E402
    validate_canonical_replay_v4,
    validate_v4_evidence,
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
    parser.add_argument("--kind", choices=("replay", "evidence"), required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = _read_object(Path(args.input_json).resolve())
    if args.kind == "replay":
        normalized = validate_canonical_replay_v4(payload)
        replay_sha = normalized["replay_semantic_sha256"]
    else:
        evidence = validate_v4_evidence(payload)
        replay_sha = str(evidence["replay_semantic_sha256"])
    report = {
        "schema_version": "factor-governance-v4-replay-validation-report.v1",
        "protocol_version": "v4",
        "status": "verified",
        "kind": args.kind,
        "replay_semantic_sha256": replay_sha,
        "complete_chain_hash_binding_verified": True,
        "positive_weight_depends_on_risk_advisor_approval": False,
        "production_apply_enabled": False,
        "registry_mutation_performed": False,
    }
    _write_private_json(Path(args.output_json).resolve(), report)
    print("factor_v4_replay_status=verified")
    print("production_apply_enabled=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
