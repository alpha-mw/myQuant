"""Dedicated CLI for the v17 protocol-v2 research-only runtime."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from quant_investor.v17_v2_contract.action_matrix import ActionMatrixError
from quant_investor.v17_v2_contract.canonical import (
    CanonicalContractError,
    load_canonical_resource,
)

from .gate import RuntimeGate, RuntimeGateError
from .service import RuntimeServiceError, analyze_mapping, verify_runtime


def _emit(payload: Any) -> None:
    sys.stdout.write(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )


def _read_input(path: str) -> dict[str, Any]:
    target = Path(path)
    try:
        raw = target.read_bytes()
    except OSError as exc:
        raise RuntimeServiceError(f"analysis input is unreadable: {target}") from exc
    try:
        payload = load_canonical_resource(raw, label="analysis input")
    except CanonicalContractError as exc:
        raise RuntimeServiceError(str(exc)) from exc
    if type(payload) is not dict:
        raise RuntimeServiceError("analysis input root must be an object")
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quant-investor-v17-v2",
        description=(
            "myQuant V17 protocol-v2 research/shadow runtime; "
            "authority=false, no broker or production cutover"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify", help="verify exact package and Phase-1 matrix")

    gate = subparsers.add_parser("gate", help="evaluate one frozen action-matrix cell")
    gate.add_argument("--action", required=True)
    gate.add_argument("--run-id", required=True)
    gate.add_argument("--version", default="ABSENT")
    gate.add_argument("--state", default="MISSING")
    gate.add_argument("--checkpoint", default="PRE_IMPORT")

    analyze = subparsers.add_parser(
        "analyze",
        help="run the pure offline deterministic pipeline from canonical JSON",
    )
    analyze.add_argument("--input", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            _emit(verify_runtime().to_wire())
            return 0
        if args.command == "gate":
            decision = RuntimeGate(Path.cwd()).classify(
                args.action,
                args.run_id,
                version=args.version,
                state=args.state,
                checkpoint=args.checkpoint,
            )
            _emit(
                {
                    **asdict(decision),
                    "outcomes": [asdict(outcome) for outcome in decision.outcomes],
                    "authority": False,
                }
            )
            return 0 if decision.allowed else 2
        if args.command == "analyze":
            result = analyze_mapping(_read_input(args.input))
            _emit(result.to_wire())
            return 0 if not result.terminal_state.startswith("HARD_STOP_") else 2
    except (
        ActionMatrixError,
        RuntimeGateError,
        RuntimeServiceError,
        TypeError,
        ValueError,
    ) as exc:
        _emit(
            {
                "authority": False,
                "error": type(exc).__name__,
                "detail": str(exc),
                "status": "BLOCKED",
            }
        )
        return 2
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
