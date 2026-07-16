#!/usr/bin/env python3
"""Build one Factor v3 baseline bootstrap plan; never apply it."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_investor.factors.factor_baseline_bootstrap import (  # noqa: E402
    FactorBaselineBootstrapError,
    build_factor_baseline_bootstrap_plan,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-path", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FactorBaselineBootstrapError("candidate manifest must be an object")
    return payload


def _write_private(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        if path.read_bytes() != raw or path.stat().st_mode & 0o777 != 0o600:
            raise FactorBaselineBootstrapError("bootstrap plan readback failed")
    finally:
        if tmp.exists():
            tmp.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plan = build_factor_baseline_bootstrap_plan(
            registry_path=args.registry_path,
            candidate_manifest=_load_object(Path(args.candidate_manifest).expanduser()),
        )
        _write_private(Path(args.output).expanduser(), plan)
    except (FactorBaselineBootstrapError, OSError, ValueError, json.JSONDecodeError) as exc:
        print("factor_baseline_bootstrap_status=blocked", file=sys.stderr)
        print(f"blocker={exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "ready_plan_only", "plan_sha256": plan["plan_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
