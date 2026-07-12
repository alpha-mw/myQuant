#!/usr/bin/env python3
"""Build an explicit, hash-bound historical factor shadow manifest.

This command reads the formal registry but never writes it.  Every
``--factor NAME=WEIGHT`` entry is bound to the raw registry-record content and
the output is a self-hashed report-only manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_investor.factors.historical_shadow import (  # noqa: E402
    build_historical_baseline_manifest,
)


def _factor_weights(values: Sequence[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for raw in values:
        name, separator, weight_text = str(raw or "").partition("=")
        name = name.strip()
        if not separator or not name or name in result:
            raise ValueError(f"invalid or duplicate --factor entry:{raw}")
        try:
            result[name] = float(weight_text)
        except ValueError as exc:
            raise ValueError(f"invalid --factor weight:{raw}") from exc
    if not result:
        raise ValueError("at least one --factor NAME=WEIGHT is required")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry-path",
        default="quant_investor/factor_registry/mined_factors.json",
    )
    parser.add_argument("--baseline-id", required=True)
    parser.add_argument("--factor", action="append", default=[], metavar="NAME=WEIGHT")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args(argv)


def _write_private_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        path.chmod(0o600)
    finally:
        if tmp.exists():
            tmp.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = build_historical_baseline_manifest(
            registry_path=args.registry_path,
            baseline_id=args.baseline_id,
            factor_weights=_factor_weights(args.factor),
        )
        output = Path(args.output_json).expanduser()
        _write_private_json(output, manifest)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"historical_factor_shadow_manifest_blocked={exc}", file=sys.stderr)
        return 2
    print(f"historical_baseline_manifest={output}")
    print(f"manifest_sha256={manifest['manifest_sha256']}")
    print("production_runtime_effect=none")
    print("formal_registry_mutated=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
